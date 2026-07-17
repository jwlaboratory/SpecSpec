#!/usr/bin/env python3
"""Adapter router — an MLP over the target's hidden states that picks which LoRA
to route a request to (polish / korean / italian / japanese / german / other).

Why this works with ZERO serving overhead: DFlash already runs the target's
prefill and extracts hidden states at layers [1,9,17,25,33] to condition the
drafter (extract_context_feature). The router consumes exactly those states,
mean-pooled over the prompt (20480-dim), so at spec-decode time the features are
already computed — routing is one tiny MLP forward.

Classes = the five multilingual LoRAs from ../finetuning/multilingual plus
"other" -> no adapter (base drafter). "other" is trained on diverse negatives
(English/French/Spanish, code, tasks, medical, financial) including deliberate
hard negatives near the adapter languages.

Stages (single H200 container, minutes total):
  1. extract — batched prefill of every prompt (train/val/test x 6 classes),
     mean-pool target hidden states at the drafter's layers -> features on volume
  2. train   — MLP 20480 -> 512 -> 6, early-stop on val accuracy
  3. eval    — held-out test accuracy, per-class + 6x6 confusion matrix

Run:
    modal run router/pipeline_router.py::run          # extract + train + eval
    modal run router/pipeline_router.py::run --skip-extract   # iterate on the MLP
    modal run router/pipeline_router.py::smoke        # tiny path validation
"""
import pathlib

import modal

LOCAL = pathlib.Path(__file__).resolve().parent           # router/
REPO = LOCAL.parent
LANG_DATA = REPO / "finetuning" / "multilingual" / "data" # lang_<l>/{train,val,test}.jsonl
OTHER_DATA = LOCAL / "data" / "other"

TARGET_MODEL = "Qwen/Qwen3-8B"
TARGET_LAYER_IDS = [1, 9, 17, 25, 33]   # == the DFlash drafter's target_layer_ids
GPU = "H200"

LANGS = ["polish", "korean", "italian", "japanese", "german"]
CLASSES = LANGS + ["other"]              # "other" -> base drafter (no adapter)

app = modal.App("dflash-adapter-router")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.9.1", "transformers==4.57.3", "accelerate>=1.0.0", "numpy")
    .env({"HF_HOME": "/cache", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_dir(str(LANG_DATA), "/root/data/langs")
    .add_local_dir(str(OTHER_DATA), "/root/data/other")
)

hf_cache = modal.Volume.from_name("dflash-hf-cache", create_if_missing=True)
work = modal.Volume.from_name("code-sql-pipeline", create_if_missing=True)
VOLS = {"/cache": hf_cache, "/work": work}

FEAT = "/work/router/features"
OUT = "/work/router"


def _read_prompts(cls, split):
    import json
    path = (f"/root/data/other/{split}.jsonl" if cls == "other"
            else f"/root/data/langs/lang_{cls}/{split}.jsonl")
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line)["prompt"])
    return out


# --------------------------------------------------------------------------- #
# 1) EXTRACT — mean-pooled target hidden states at the drafter's layers
# --------------------------------------------------------------------------- #
@app.function(gpu=GPU, image=image, timeout=2 * 3600, volumes=VOLS)
def extract(limit: int = 0, batch: int = 32, max_len: int = 512):
    import os
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda").eval()

    os.makedirs(FEAT, exist_ok=True)
    summary = {}
    for split in ("train", "val", "test"):
        feats, labels = [], []
        t0 = time.time()
        for ci, cls in enumerate(CLASSES):
            prompts = _read_prompts(cls, split)
            if limit:
                prompts = prompts[:limit]
            texts = [tok.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            ) for p in prompts]
            for i in range(0, len(texts), batch):
                enc = tok(texts[i:i + batch], return_tensors="pt", padding=True,
                          truncation=True, max_length=max_len).to("cuda")
                with torch.inference_mode():
                    hs = target(**enc, use_cache=False, output_hidden_states=True).hidden_states
                # same feature the drafter conditions on: concat layers, +1 offset
                # (hidden_states[0] is the embedding layer)
                sel = torch.cat([hs[lid + 1] for lid in TARGET_LAYER_IDS], dim=-1)  # (B,S,20480)
                mask = enc.attention_mask.unsqueeze(-1)                              # (B,S,1)
                pooled = (sel * mask).sum(1) / mask.sum(1).clamp_min(1)              # (B,20480)
                feats.append(pooled.float().cpu())
                labels += [ci] * pooled.shape[0]
            print(f"[extract:{split}:{cls}] {len(prompts)} prompts "
                  f"({time.time()-t0:.0f}s)", flush=True)
        X = torch.cat(feats).to(torch.float16)
        y = torch.tensor(labels, dtype=torch.long)
        torch.save({"X": X, "y": y, "classes": CLASSES,
                    "layer_ids": TARGET_LAYER_IDS, "pool": "mean"},
                   f"{FEAT}/{split}.pt")
        summary[split] = {"n": int(y.numel()), "dim": int(X.shape[1]),
                          "sec": round(time.time() - t0, 1)}
        print(f"[extract:{split}] saved {summary[split]}", flush=True)
    work.commit()
    return summary


# --------------------------------------------------------------------------- #
# 2+3) TRAIN + EVAL — small MLP, early-stop on val, confusion matrix on test
# --------------------------------------------------------------------------- #
@app.function(gpu=GPU, image=image, timeout=3600, volumes=VOLS)
def train_and_eval(hidden: int = 512, dropout: float = 0.1, lr: float = 1e-3,
                   epochs: int = 40, batch: int = 256, seed: int = 0):
    import json

    import torch
    import torch.nn as nn

    work.reload()
    torch.manual_seed(seed)
    dev = "cuda"

    tr = torch.load(f"{FEAT}/train.pt")
    va = torch.load(f"{FEAT}/val.pt")
    te = torch.load(f"{FEAT}/test.pt")
    classes = tr["classes"]
    dim = tr["X"].shape[1]
    n_cls = len(classes)

    def norm_stats(X):
        mu = X.float().mean(0)
        sd = X.float().std(0).clamp_min(1e-6)
        return mu, sd

    mu, sd = norm_stats(tr["X"])

    def prep(d):
        return ((d["X"].float() - mu) / sd).to(dev), d["y"].to(dev)

    Xtr, ytr = prep(tr); Xva, yva = prep(va); Xte, yte = prep(te)

    mlp = nn.Sequential(
        nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, n_cls),
    ).to(dev)
    opt = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()

    @torch.no_grad()
    def acc(X, y):
        mlp.eval()
        return float((mlp(X).argmax(-1) == y).float().mean())

    best_va, best_state, patience = 0.0, None, 0
    for ep in range(epochs):
        mlp.train()
        perm = torch.randperm(Xtr.shape[0], device=dev)
        for i in range(0, Xtr.shape[0], batch):
            idx = perm[i:i + batch]
            loss = lossf(mlp(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        va_acc = acc(Xva, yva)
        if va_acc > best_va:
            best_va, patience = va_acc, 0
            best_state = {k: v.detach().clone() for k, v in mlp.state_dict().items()}
        else:
            patience += 1
        if ep % 5 == 0 or patience == 0:
            print(f"[router] ep{ep} train_loss={loss.item():.4f} val_acc={va_acc:.4f} "
                  f"(best {best_va:.4f})", flush=True)
        if patience >= 8:
            print(f"[router] early stop at ep{ep}", flush=True)
            break
    mlp.load_state_dict(best_state)

    # ---- test eval + confusion matrix ----
    mlp.eval()
    with torch.no_grad():
        pred = mlp(Xte).argmax(-1)
    test_acc = float((pred == yte).float().mean())
    conf = torch.zeros(n_cls, n_cls, dtype=torch.long)
    for t, p in zip(yte.cpu(), pred.cpu()):
        conf[t, p] += 1
    per_class = {classes[i]: float(conf[i, i] / conf[i].sum().clamp_min(1))
                 for i in range(n_cls)}

    # ---- save router checkpoint (everything inference needs) ----
    ckpt = {"state_dict": {k: v.cpu() for k, v in mlp.state_dict().items()},
            "classes": classes, "layer_ids": tr["layer_ids"], "pool": tr["pool"],
            "hidden": hidden, "dim": dim, "mu": mu.cpu(), "sd": sd.cpu(),
            "val_acc": best_va, "test_acc": test_acc}
    torch.save(ckpt, f"{OUT}/router_mlp.pt")

    # ---- report ----
    md = ["# Adapter router — MLP on target hidden states\n",
          f"Features: mean-pooled Qwen3-8B hidden states at layers {tr['layer_ids']} "
          f"({dim}-dim — the same states the DFlash drafter already consumes, so "
          "routing costs one tiny MLP forward at serve time).\n",
          f"MLP {dim}→{hidden}→{n_cls} · train {Xtr.shape[0]} / val {Xva.shape[0]} / "
          f"test {Xte.shape[0]}\n",
          f"**Val accuracy: {best_va*100:.1f}%  ·  Test accuracy: {test_acc*100:.1f}%**\n",
          "## Per-class test accuracy\n",
          "| class | accuracy | routes to |", "|---|--:|---|"]
    for c in classes:
        md.append(f"| {c} | {per_class[c]*100:.1f}% | "
                  f"{'base drafter (no adapter)' if c == 'other' else c + ' LoRA'} |")
    md += ["\n## Confusion matrix (rows = true, cols = predicted)\n",
           "| | " + " | ".join(classes) + " |",
           "|---|" + "--:|" * n_cls]
    for i, c in enumerate(classes):
        md.append(f"| **{c}** | " + " | ".join(str(int(x)) for x in conf[i]) + " |")
    md_text = "\n".join(md) + "\n"
    with open(f"{OUT}/router_report.md", "w") as f:
        f.write(md_text)
    with open(f"{OUT}/router_confusion.json", "w") as f:
        json.dump({"classes": classes, "confusion": conf.tolist(),
                   "per_class": per_class, "val_acc": best_va,
                   "test_acc": test_acc}, f, indent=2)
    work.commit()
    print("\n" + md_text)
    return {"val_acc": best_va, "test_acc": test_acc, "per_class": per_class,
            "report": md_text}


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
@app.local_entrypoint()
def run(skip_extract: bool = False, hidden: int = 512, epochs: int = 40):
    import json

    if not skip_extract:
        print("=== extract features ===")
        print(json.dumps(extract.remote(), indent=2))
    print("=== train + eval router ===")
    out = train_and_eval.remote(hidden=hidden, epochs=epochs)
    print(f"val={out['val_acc']*100:.1f}%  test={out['test_acc']*100:.1f}%")


@app.local_entrypoint()
def smoke():
    import json

    print("=== SMOKE extract (12 prompts/class) ===")
    print(json.dumps(extract.remote(limit=12), indent=2))
    print("=== SMOKE train+eval ===")
    out = train_and_eval.remote(epochs=10)
    print(f"val={out['val_acc']*100:.1f}%  test={out['test_acc']*100:.1f}%")
