#!/usr/bin/env python3
"""40-way WildChat language router for the language-LoRA experiment.

This retrains the adapter router on the actual `new/exp1-language` prompt
splits: one class per WildChat language lane, no `other` bucket.

Features are the same mean-pooled Qwen3-8B hidden states the DFlash drafter
already consumes at serve time: target layers [1, 9, 17, 25, 33], concatenated to
20480 dimensions.

Run:
    modal run router/pipeline_router_40.py::smoke
    modal run --detach router/pipeline_router_40.py::run
    modal run router/pipeline_router_40.py::results
"""
from __future__ import annotations

import pathlib

import modal

LOCAL = pathlib.Path(__file__).resolve().parent

TARGET_MODEL = "Qwen/Qwen3-8B"
TARGET_LAYER_IDS = [1, 9, 17, 25, 33]
GPU = "H200"

# Same 40-language list as new/exp1-language/pipeline.py.
LANGS = [
    "English", "Russian", "Chinese", "French", "Vietnamese", "Yoruba", "Arabic",
    "Indonesian", "Spanish", "Portuguese", "German", "Persian", "Tagalog",
    "Turkish", "Korean", "Italian", "Maori", "Sotho", "Polish", "Latin",
    "Japanese", "Serbian", "Ukrainian", "Malay", "Dutch", "Esperanto",
    "Romanian", "Hungarian", "Swedish", "Somali", "Estonian", "Tswana",
    "Bulgarian", "Finnish", "Catalan", "Bokmal", "Hebrew", "Welsh", "Hindi",
    "Nynorsk",
]

app = modal.App("dflash-language-router-40")

hf_cache = modal.Volume.from_name("dflash-hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("exp1-language-hidden", create_if_missing=True)
VOLS = {"/cache": hf_cache, "/data": data_vol}

PROMPTS = "/data/prompts"
OUT = "/data/router40"
FEAT = f"{OUT}/features"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.9.1", "transformers==4.57.3", "accelerate>=1.0.0", "numpy")
    .env({"HF_HOME": "/cache", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
)


def _parse_langs(langs: str | None) -> list[str]:
    if not langs:
        return LANGS
    wanted = [x.strip() for x in langs.split(",") if x.strip()]
    by_lower = {x.lower(): x for x in LANGS}
    out = []
    for lang in wanted:
        resolved = by_lower.get(lang.lower())
        if resolved is None:
            raise ValueError(f"unknown language {lang!r}")
        out.append(resolved)
    return out


def _read_prompts(lang: str, split: str) -> list[str]:
    import json

    out = []
    with open(f"{PROMPTS}/{lang}/{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line)["prompt"])
    return out


@app.function(image=image, timeout=300, volumes=VOLS)
def check_inputs(langs: list[str]) -> dict:
    import os

    counts = {}
    missing = []
    for lang in langs:
        counts[lang] = {}
        for split in ("train", "val", "test"):
            path = f"{PROMPTS}/{lang}/{split}.jsonl"
            if not os.path.exists(path):
                missing.append(path)
                continue
            n = sum(1 for line in open(path, encoding="utf-8") if line.strip())
            counts[lang][split] = n
    if missing:
        raise FileNotFoundError(missing)
    return {"langs": langs, "n_classes": len(langs), "counts": counts}


@app.function(gpu=GPU, image=image, timeout=6 * 3600, volumes=VOLS,
              memory=49152, retries=1)
def extract(langs: list[str] = LANGS, limit: int = 0, batch: int = 32,
            max_len: int = 512) -> dict:
    import os
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    target = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda").eval()

    os.makedirs(FEAT, exist_ok=True)
    summary = {"classes": langs, "splits": {}, "limit": limit,
               "batch": batch, "max_len": max_len}

    for split in ("train", "val", "test"):
        feats, labels = [], []
        split_t0 = time.time()
        for ci, lang in enumerate(langs):
            prompts = _read_prompts(lang, split)
            if limit:
                prompts = prompts[:limit]
            texts = [
                tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for prompt in prompts
            ]
            for i in range(0, len(texts), batch):
                enc = tok(
                    texts[i:i + batch],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                ).to("cuda")
                with torch.inference_mode():
                    hs = target(**enc, use_cache=False,
                                output_hidden_states=True).hidden_states
                    sel = torch.cat([hs[lid + 1] for lid in TARGET_LAYER_IDS], dim=-1)
                    mask = enc.attention_mask.unsqueeze(-1)
                    pooled = (sel * mask).sum(1) / mask.sum(1).clamp_min(1)
                feats.append(pooled.float().cpu())
                labels += [ci] * pooled.shape[0]
                del enc, hs, sel, mask, pooled
            print(f"[extract:{split}:{lang}] {len(prompts)} prompts", flush=True)

        X = torch.cat(feats).to(torch.float16)
        y = torch.tensor(labels, dtype=torch.long)
        path = f"{FEAT}/{split}.pt"
        torch.save({
            "X": X,
            "y": y,
            "classes": langs,
            "layer_ids": TARGET_LAYER_IDS,
            "pool": "mean",
        }, path)
        summary["splits"][split] = {
            "n": int(y.numel()),
            "dim": int(X.shape[1]),
            "sec": round(time.time() - split_t0, 1),
            "path": path,
        }
        print(f"[extract:{split}] saved {summary['splits'][split]}", flush=True)
        del feats, labels, X, y

    data_vol.commit()
    return summary


@app.function(gpu=GPU, image=image, timeout=2 * 3600, volumes=VOLS,
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

    tr = torch.load(f"{FEAT}/train.pt")
    va = torch.load(f"{FEAT}/val.pt")
    te = torch.load(f"{FEAT}/test.pt")
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
            print(f"[router40] ep={ep} loss={last_loss:.4f} "
                  f"val={va_acc:.4f} best={best_va:.4f}", flush=True)
        if patience >= 8:
            print(f"[router40] early stop at epoch {ep}", flush=True)
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
    torch.save(ckpt, f"{OUT}/router40_mlp.pt")

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
    }
    with open(f"{OUT}/router40_confusion.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    md = [
        "# 40-language adapter router\n",
        f"Features: mean-pooled Qwen3-8B hidden states at layers {tr['layer_ids']} "
        f"({dim}-dim, same DFlash target context feature).\n",
        f"MLP {dim}->{hidden}->{n_cls} · train {Xtr.shape[0]} / "
        f"val {Xva.shape[0]} / test {Xte.shape[0]}\n",
        f"**Val accuracy: {best_va*100:.2f}% · Test accuracy: {test_acc*100:.2f}%**\n",
        "## Per-class test accuracy\n",
        "| language | accuracy | test n |",
        "|---|---:|---:|",
    ]
    for i, c in enumerate(classes):
        md.append(f"| {c} | {per_class[c]*100:.2f}% | {int(conf[i].sum())} |")
    md += [
        "\n## Confusion matrix\n",
        "| | " + " | ".join(classes) + " |",
        "|---|" + "--:|" * n_cls,
    ]
    for i, c in enumerate(classes):
        md.append(f"| **{c}** | " + " | ".join(str(int(x)) for x in conf[i]) + " |")
    md_text = "\n".join(md) + "\n"
    with open(f"{OUT}/router40_report.md", "w", encoding="utf-8") as f:
        f.write(md_text)

    data_vol.commit()
    print("\n" + md_text)
    return result


@app.function(image=image, timeout=300, volumes=VOLS)
def read_results() -> dict:
    import json
    import os

    path = f"{OUT}/router40_confusion.json"
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
def run(langs: str = "", skip_extract: bool = False, limit: int = 0,
        batch: int = 32, max_len: int = 512, hidden: int = 512,
        epochs: int = 40):
    import json

    parsed = _parse_langs(langs)
    print(json.dumps(check_inputs.remote(parsed), indent=2))
    if not skip_extract:
        print("=== extract 40-way router features ===")
        print(json.dumps(extract.remote(parsed, limit=limit, batch=batch,
                                        max_len=max_len), indent=2))
    print("=== train + eval 40-way router ===")
    out = train_and_eval.remote(hidden=hidden, epochs=epochs)
    print(f"val={out['val_acc']*100:.2f}% test={out['test_acc']*100:.2f}%")


@app.local_entrypoint()
def smoke(langs: str = "English,Korean,Polish", limit: int = 12):
    import json

    parsed = _parse_langs(langs)
    print(json.dumps(check_inputs.remote(parsed), indent=2))
    print("=== smoke extract ===")
    print(json.dumps(extract.remote(parsed, limit=limit, batch=4,
                                    max_len=256), indent=2))
    print("=== smoke train + eval ===")
    out = train_and_eval.remote(hidden=128, epochs=10, batch=32)
    print(f"val={out['val_acc']*100:.2f}% test={out['test_acc']*100:.2f}%")


@app.local_entrypoint()
def results():
    import json

    print(json.dumps(read_results.remote(), indent=2))
