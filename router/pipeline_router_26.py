#!/usr/bin/env python3
"""26-way clean WildChat language router.

This trains on the `min_train=1000` language subset by reusing the hidden-state
features already extracted by `pipeline_router_40.py` under
`/data/router40/features`. No target-model prefill is needed.

Run:
    modal run router/pipeline_router_26.py::run
    modal run router/pipeline_router_26.py::results
"""
from __future__ import annotations

import modal

LANGS = [
    "English", "Russian", "Chinese", "French", "Vietnamese", "Yoruba", "Arabic",
    "Indonesian", "Spanish", "Portuguese", "German", "Persian", "Tagalog",
    "Turkish", "Korean", "Italian", "Polish", "Latin", "Japanese", "Ukrainian",
    "Malay", "Dutch", "Esperanto", "Romanian", "Hungarian", "Swedish",
]

app = modal.App("dflash-language-router-26")

data_vol = modal.Volume.from_name("exp1-language-hidden", create_if_missing=True)
VOLS = {"/data": data_vol}

IN_FEAT = "/data/router40/features"
OUT = "/data/router26"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.9.1", "numpy")
)


def _filter_split(d, keep_classes: list[str]):
    import torch

    old_classes = list(d["classes"])
    old_to_new = {old_classes.index(cls): i for i, cls in enumerate(keep_classes)}
    keep_old = set(old_to_new)
    mask = torch.tensor([int(y) in keep_old for y in d["y"].tolist()], dtype=torch.bool)
    y_old = d["y"][mask]
    y = torch.tensor([old_to_new[int(v)] for v in y_old.tolist()], dtype=torch.long)
    return {
        "X": d["X"][mask],
        "y": y,
        "classes": keep_classes,
        "layer_ids": d["layer_ids"],
        "pool": d["pool"],
    }


@app.function(image=image, timeout=300, volumes=VOLS)
def check_features() -> dict:
    import os

    missing = [f"{IN_FEAT}/{split}.pt" for split in ("train", "val", "test")
               if not os.path.exists(f"{IN_FEAT}/{split}.pt")]
    if missing:
        raise FileNotFoundError(missing)
    return {"features": IN_FEAT, "out": OUT, "classes": LANGS, "n_classes": len(LANGS)}


@app.function(gpu="H200", image=image, timeout=2 * 3600, volumes=VOLS,
              memory=49152, retries=1)
def train_and_eval(hidden: int = 512, dropout: float = 0.1, lr: float = 1e-3,
                   epochs: int = 40, batch: int = 512, seed: int = 0) -> dict:
    import json
    import os

    import torch
    import torch.nn as nn

    data_vol.reload()
    torch.manual_seed(seed)
    dev = "cuda"

    tr = _filter_split(torch.load(f"{IN_FEAT}/train.pt"), LANGS)
    va = _filter_split(torch.load(f"{IN_FEAT}/val.pt"), LANGS)
    te = _filter_split(torch.load(f"{IN_FEAT}/test.pt"), LANGS)
    classes = list(tr["classes"])
    dim = int(tr["X"].shape[1])
    n_cls = len(classes)

    mu = tr["X"].float().mean(0)
    sd = tr["X"].float().std(0).clamp_min(1e-6)

    def prep(d):
        return ((d["X"].float() - mu) / sd).to(dev), d["y"].to(dev)

    Xtr, ytr = prep(tr)
    Xva, yva = prep(va)
    Xte, yte = prep(te)

    mlp = nn.Sequential(
        nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, n_cls),
    ).to(dev)
    opt = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()

    @torch.no_grad()
    def accuracy(X, y):
        mlp.eval()
        return float((mlp(X).argmax(-1) == y).float().mean())

    best_va = -1.0
    best_state = None
    patience = 0
    history = []
    for ep in range(epochs):
        mlp.train()
        perm = torch.randperm(Xtr.shape[0], device=dev)
        last_loss = None
        for i in range(0, Xtr.shape[0], batch):
            idx = perm[i:i + batch]
            loss = lossf(mlp(Xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu())
        va_acc = accuracy(Xva, yva)
        history.append({"epoch": ep, "train_loss": last_loss, "val_acc": va_acc})
        if va_acc > best_va:
            best_va = va_acc
            patience = 0
            best_state = {k: v.detach().clone() for k, v in mlp.state_dict().items()}
        else:
            patience += 1
        if ep % 5 == 0 or patience == 0:
            print(f"[router26] ep={ep} loss={last_loss:.4f} "
                  f"val={va_acc:.4f} best={best_va:.4f}", flush=True)
        if patience >= 8:
            print(f"[router26] early stop at epoch {ep}", flush=True)
            break

    assert best_state is not None
    mlp.load_state_dict(best_state)
    mlp.eval()

    with torch.no_grad():
        pred = mlp(Xte).argmax(-1)
    test_acc = float((pred == yte).float().mean())
    conf = torch.zeros(n_cls, n_cls, dtype=torch.long)
    for t, p in zip(yte.cpu(), pred.cpu()):
        conf[t, p] += 1
    per_class = {
        classes[i]: float(conf[i, i] / conf[i].sum().clamp_min(1))
        for i in range(n_cls)
    }

    os.makedirs(OUT, exist_ok=True)
    ckpt = {
        "state_dict": {k: v.cpu() for k, v in mlp.state_dict().items()},
        "classes": classes,
        "layer_ids": tr["layer_ids"],
        "pool": tr["pool"],
        "hidden": hidden,
        "dim": dim,
        "mu": mu.cpu(),
        "sd": sd.cpu(),
        "val_acc": best_va,
        "test_acc": test_acc,
    }
    torch.save(ckpt, f"{OUT}/router26_mlp.pt")

    result = {
        "classes": classes,
        "confusion": conf.tolist(),
        "per_class": per_class,
        "val_acc": best_va,
        "test_acc": test_acc,
        "history": history,
        "train_n": int(Xtr.shape[0]),
        "val_n": int(Xva.shape[0]),
        "test_n": int(Xte.shape[0]),
        "dim": dim,
        "hidden": hidden,
        "source_features": IN_FEAT,
    }
    with open(f"{OUT}/router26_confusion.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    md = [
        "# 26-language adapter router\n",
        f"Features: filtered from `{IN_FEAT}`; mean-pooled Qwen3-8B hidden states "
        f"({dim}-dim).\n",
        f"MLP {dim}->{hidden}->{n_cls} · train {Xtr.shape[0]} / "
        f"val {Xva.shape[0]} / test {Xte.shape[0]}\n",
        f"**Val accuracy: {best_va*100:.2f}% · Test accuracy: {test_acc*100:.2f}%**\n",
        "## Per-class test accuracy\n",
        "| language | accuracy | test n |",
        "|---|---:|---:|",
    ]
    for i, c in enumerate(classes):
        md.append(f"| {c} | {per_class[c]*100:.2f}% | {int(conf[i].sum())} |")
    md_text = "\n".join(md) + "\n"
    with open(f"{OUT}/router26_report.md", "w", encoding="utf-8") as f:
        f.write(md_text)

    data_vol.commit()
    print("\n" + md_text)
    return result


@app.function(image=image, timeout=300, volumes=VOLS)
def read_results() -> dict:
    import json
    import os

    path = f"{OUT}/router26_confusion.json"
    if not os.path.exists(path):
        return {"done": False, "path": path}
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return {
        "done": True,
        "path": path,
        "val_acc": d["val_acc"],
        "test_acc": d["test_acc"],
        "n_classes": len(d["classes"]),
        "train_n": d["train_n"],
        "val_n": d["val_n"],
        "test_n": d["test_n"],
    }


@app.local_entrypoint()
def run(hidden: int = 512, epochs: int = 40):
    import json

    print(json.dumps(check_features.remote(), indent=2))
    out = train_and_eval.remote(hidden=hidden, epochs=epochs)
    print(f"val={out['val_acc']*100:.2f}% test={out['test_acc']*100:.2f}%")


@app.local_entrypoint()
def results():
    import json

    print(json.dumps(read_results.remote(), indent=2))
