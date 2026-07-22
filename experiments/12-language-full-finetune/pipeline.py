#!/usr/bin/env python3
"""Language-specific full fine-tune baseline for DFlash.

This reuses the WildChat language data from `new/exp1-language`:

  * train full DFlash weights on frozen target-hidden shards for weak languages;
  * compare against existing base / own-LoRA / combined-LoRA held-out results;
  * write a compact report for the blog.
"""
from __future__ import annotations

import pathlib

import modal

LOCAL = pathlib.Path(__file__).resolve().parent
ROOT = LOCAL.parent.parent
LORA = ROOT / "lib" / "lora.py"
ONLINE = ROOT / "lib" / "online_dflash.py"
SPEC_PATCH = ROOT / "lib" / "spec_patch.py"

DRAFT_MODEL = "z-lab/Qwen3-8B-DFlash-b16"
TARGET_MODEL = "Qwen/Qwen3-8B"
GPU_TRAIN = "A100-40GB"
GPU_BENCH = "H200"
VARIANTS = ["base", "own", "combined", "full"]
DEFAULT_LANGS = ["Polish", "Hungarian", "Korean", "Hebrew", "Dutch"]

app = modal.App("language-full-finetune")

hf_cache = modal.Volume.from_name("dflash-hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("exp1-language-hidden")
VOLS = {"/cache": hf_cache, "/data": data_vol}

PROMPTS = "/data/prompts"
SHARDS = "/data/shards"
HEAD_DUMP = "/data/target_head_embed.pt"
MODELS = "/data/models"
EXP = "/data/exp12_language_full_finetune"
FULL_MODELS = f"{EXP}/models"
RESULTS = f"{EXP}/results"

cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "torch==2.9.1")
    .env({"HF_HOME": "/cache"})
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.9.1", "transformers==4.57.3", "accelerate>=1.0.0",
                 "datasets>=3.0.0", "numpy")
    .env({
        "HF_HOME": "/cache",
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    .add_local_file(str(ONLINE), "/root/online_dflash.py")
    .add_local_file(str(LORA), "/root/lora.py")
    .add_local_file(str(SPEC_PATCH), "/root/spec_patch.py")
)


def _chat_text(tok, prompt: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _read_prompts(lang: str, split: str) -> list[str]:
    import json

    out = []
    with open(f"{PROMPTS}/{lang}/{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line)["prompt"])
    return out


def _shard_paths(langs: list[str], split: str) -> list[str]:
    import json
    import os

    paths = []
    for lang in langs:
        d = f"{SHARDS}/{lang}/{split}"
        with open(f"{d}/manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)
        paths += [os.path.join(d, s) for s in manifest["shards"]]
    return paths


def _load_draft():
    import torch
    from transformers import AutoModel

    draft = AutoModel.from_pretrained(
        DRAFT_MODEL,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    return draft


def _load_target():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    for p in target.parameters():
        p.requires_grad_(False)
    return tok, target


def _load_head_embed():
    import torch

    dump = torch.load(HEAD_DUMP, map_location="cpu")
    vocab, hidden = dump["lm_head.weight"].shape
    lm_head = torch.nn.Linear(hidden, vocab, bias=False, dtype=torch.bfloat16, device="cuda")
    lm_head.weight.data.copy_(dump["lm_head.weight"])
    embed = torch.nn.Embedding(vocab, hidden, dtype=torch.bfloat16, device="cuda")
    embed.weight.data.copy_(dump["embed_tokens.weight"])
    for p in list(lm_head.parameters()) + list(embed.parameters()):
        p.requires_grad_(False)
    return lm_head, embed


def _make_batch(records, *, pad_token_id: int, max_seq_len: int):
    import torch

    if max_seq_len:
        records = [
            {**r, "input_ids": r["input_ids"][:max_seq_len], "hidden": r["hidden"][:max_seq_len]}
            for r in records
        ]
    bsz = len(records)
    seqlen = max(r["input_ids"].shape[0] for r in records)
    n_ctx = records[0]["hidden"].shape[1]
    input_ids = torch.full((bsz, seqlen), pad_token_id, dtype=torch.long)
    hidden = torch.zeros((bsz, seqlen, n_ctx), dtype=torch.bfloat16)
    loss_mask = torch.zeros((bsz, seqlen), dtype=torch.float)
    for i, rec in enumerate(records):
        n = rec["input_ids"].shape[0]
        input_ids[i, :n] = rec["input_ids"].long()
        hidden[i, :n] = rec["hidden"]
        loss_mask[i, min(rec["prompt_len"], n):n] = 1.0
    return input_ids.to("cuda"), hidden.to("cuda"), loss_mask.to("cuda")


def _load_lora_into(draft, sd):
    import torch
    from lora import LoRALinear

    named = dict(draft.named_modules())
    for name, entry in sd.items():
        module = named.get(name)
        assert isinstance(module, LoRALinear), f"missing LoRALinear at {name}"
        with torch.no_grad():
            module.A.copy_(entry["A"].to(module.A.device, module.A.dtype))
            module.B.copy_(entry["B"].to(module.B.device, module.B.dtype))
        module.scaling = float(entry["scaling"])


@app.function(image=cpu_image, timeout=600, volumes=VOLS)
def check_inputs(langs: list[str]) -> dict:
    import os

    available = {
        name.lower(): name
        for name in os.listdir(PROMPTS)
        if os.path.isdir(f"{PROMPTS}/{name}")
    }
    resolved = [available.get(lang.lower(), lang) for lang in langs]
    missing = {}
    for lang in resolved:
        needed = [
            f"{PROMPTS}/{lang}/test.jsonl",
            f"{SHARDS}/{lang}/train/manifest.json",
            f"{SHARDS}/{lang}/val/manifest.json",
            f"{MODELS}/{lang}_lora.pt",
            f"{MODELS}/combined_lora.pt",
            HEAD_DUMP,
        ]
        absent = [p for p in needed if not os.path.exists(p)]
        if absent:
            missing[lang] = absent
    assert not missing, f"missing inputs: {missing}"
    return {"langs": resolved, "n": len(resolved)}


@app.function(gpu=GPU_TRAIN, image=image, timeout=24 * 3600, volumes=VOLS,
              max_containers=5, retries=1, memory=32768)
def train_full(
    lang: str,
    epochs: int = 3,
    lr: float = 1e-5,
    batch_size: int = 4,
    num_anchors: int = 48,
    max_seq_len: int = 512,
    val_every: int = 100,
    max_steps: int = 0,
    seed: int = 0,
):
    import os
    import random
    import sys
    import time

    import torch
    from transformers import AutoTokenizer

    sys.path.insert(0, "/root")
    from online_dflash import OnlineDFlashModel

    random.seed(seed)
    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    draft = _load_draft()
    for p in draft.parameters():
        p.requires_grad_(True)
    trainable = [p for p in draft.parameters() if p.requires_grad]
    lm_head, embed = _load_head_embed()
    online = OnlineDFlashModel(
        draft_model=draft,
        target_lm_head=lm_head,
        target_embed_tokens=embed,
        mask_token_id=draft.mask_token_id,
        block_size=draft.block_size,
        attention_backend="sdpa",
        num_anchors=num_anchors,
        loss_decay_gamma=7.0,
        loss_type="dflash",
    )
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    block_size = draft.block_size

    train_paths = _shard_paths([lang], "train")
    val_records = []
    for path in _shard_paths([lang], "val"):
        val_records += torch.load(path, map_location="cpu")
        if len(val_records) >= 60:
            break
    val_records = [
        r for r in val_records[:80]
        if min(r["input_ids"].shape[0], max_seq_len) - r["prompt_len"] > 2 * block_size
    ][:60]

    def run_batch(records, *, train: bool):
        ids, hidden, mask = _make_batch(records, pad_token_id=tok.pad_token_id,
                                        max_seq_len=max_seq_len)
        anchorable = mask[:, : max(mask.shape[1] - block_size, 0) + 1].sum(dim=1)
        keep = anchorable > block_size + 1
        if keep.sum() == 0:
            return None
        ids, hidden, mask = ids[keep], hidden[keep], mask[keep]
        try:
            loss, acc = online(input_ids=ids, hidden_states=hidden, loss_mask=mask)[:2]
        except ValueError:
            return None
        if train:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
        return float(loss.item()), float(acc.item())

    @torch.no_grad()
    def validate():
        online.eval()
        losses, accs = [], []
        for i in range(0, len(val_records), batch_size):
            rec = run_batch(val_records[i:i + batch_size], train=False)
            if rec:
                losses.append(rec[0])
                accs.append(rec[1])
        online.train()
        return (sum(losses) / len(losses), sum(accs) / len(accs)) if losses else (None, None)

    rng = random.Random(seed)
    log = {
        "lang": lang,
        "mode": "full",
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "max_steps": max_steps,
        "trainable_params": sum(p.numel() for p in trainable),
        "val": [],
    }
    online.train()
    t0 = time.time()
    step = 0
    log["val_initial"] = validate()
    print(f"[train_full:{lang}] params={log['trainable_params']:,} "
          f"initial={log['val_initial']}", flush=True)
    for ep in range(epochs):
        paths = list(train_paths)
        rng.shuffle(paths)
        for path in paths:
            recs = torch.load(path, map_location="cpu")
            starts = list(range(0, len(recs), batch_size))
            rng.shuffle(starts)
            for start in starts:
                out = run_batch(recs[start:start + batch_size], train=True)
                if out is None:
                    continue
                step += 1
                if step % 25 == 0:
                    print(f"[train_full:{lang}] ep={ep} step={step} "
                          f"loss={out[0]:.4f} acc={out[1]:.4f}", flush=True)
                if step % val_every == 0:
                    val = validate()
                    log["val"].append({"step": step, "loss": val[0], "acc": val[1]})
                    print(f"[train_full:{lang}] val step={step} {val}", flush=True)
                if max_steps and step >= max_steps:
                    break
            del recs
            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break
    log["val_final"] = validate()
    os.makedirs(FULL_MODELS, exist_ok=True)
    path = f"{FULL_MODELS}/{lang}_full.pt"
    torch.save({k: v.detach().to(torch.bfloat16).cpu() for k, v in draft.state_dict().items()}, path)
    data_vol.commit()
    log["saved"] = path
    log["train_seconds"] = round(time.time() - t0, 1)
    print(f"[train_full:{lang}] saved {path} final={log['val_final']}", flush=True)
    return log


@app.function(gpu=GPU_BENCH, image=image, timeout=3 * 3600, volumes=VOLS,
              max_containers=20, retries=1)
def bench(
    lang: str,
    variant: str,
    max_new_tokens: int = 256,
    limit: int = 100,
    warmup: int = 2,
):
    import json
    import os
    import sys
    import time

    import torch

    sys.path.insert(0, "/root")
    from spec_patch import make_instrumented_spec_generate

    assert variant in VARIANTS
    tok, target = _load_target()
    draft = _load_draft()

    if variant == "full":
        sd = torch.load(f"{FULL_MODELS}/{lang}_full.pt", map_location="cuda")
        missing, unexpected = draft.load_state_dict(sd, strict=False)
        print(f"[bench:{lang}:full] loaded missing={len(missing)} "
              f"unexpected={len(unexpected)}", flush=True)
    elif variant in ("own", "combined"):
        from lora import inject_lora

        inject_lora(draft, rank=16, alpha=32,
                    target_modules=("q_proj", "k_proj", "v_proj", "o_proj"))
        draft.to("cuda", dtype=torch.bfloat16)
        adapter = f"{MODELS}/{lang}_lora.pt" if variant == "own" else f"{MODELS}/combined_lora.pt"
        _load_lora_into(draft, torch.load(adapter, map_location="cuda"))
        print(f"[bench:{lang}:{variant}] loaded {adapter}", flush=True)

    draft.spec_generate = make_instrumented_spec_generate(draft)
    block_size = draft.block_size
    stop_ids = [tok.eos_token_id]
    prompts = _read_prompts(lang, "test")[:limit]

    def build_ids(prompt: str):
        return tok([_chat_text(tok, prompt)], return_tensors="pt").input_ids.to("cuda")

    def first_stop(ids):
        for i, token in enumerate(ids):
            if token in stop_ids:
                return ids[:i + 1]
        return ids

    warm = build_ids(prompts[0])
    for _ in range(warmup):
        with torch.inference_mode():
            draft.spec_generate(target=target, input_ids=warm, max_new_tokens=min(64, max_new_tokens),
                                stop_token_ids=stop_ids, temperature=0.0)
    torch.cuda.synchronize()

    os.makedirs(RESULTS, exist_ok=True)
    out_path = f"{RESULTS}/{lang}_{variant}.jsonl"
    recs = []
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, prompt in enumerate(prompts):
            input_ids = build_ids(prompt)
            n_in = input_ids.shape[1]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                out_ids, committed = draft.spec_generate(
                    target=target,
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    stop_token_ids=stop_ids,
                    temperature=0.0,
                )
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            spec_gen = first_stop(out_ids[0, n_in:].tolist())
            steps = len(committed)
            generated = sum(committed)
            accepted = sum(c - 1 for c in committed)
            proposed = steps * (block_size - 1)
            rec = {
                "lang": lang,
                "variant": variant,
                "prompt_idx": idx,
                "num_input_tokens": n_in,
                "forward_steps": steps,
                "num_generated_tokens": generated,
                "emitted_tokens": len(spec_gen),
                "accepted_draft_tokens": accepted,
                "proposed_draft_tokens": proposed,
                "acceptance_rate": accepted / proposed if proposed else 0.0,
                "mean_accept_length": generated / steps if steps else 0.0,
                "spec_seconds": dt,
                "spec_tok_s": len(spec_gen) / dt if dt else 0.0,
            }
            recs.append(rec)
            f.write(json.dumps(rec) + "\n")
            if (idx + 1) % 20 == 0:
                print(f"[bench:{lang}:{variant}] {idx+1}/{len(prompts)}", flush=True)
    data_vol.commit()
    accepted = sum(r["accepted_draft_tokens"] for r in recs)
    proposed = sum(r["proposed_draft_tokens"] for r in recs)
    generated = sum(r["num_generated_tokens"] for r in recs)
    steps = sum(r["forward_steps"] for r in recs)
    return {
        "lang": lang,
        "variant": variant,
        "n": len(recs),
        "acceptance_rate": accepted / proposed if proposed else 0.0,
        "mean_accept_length": generated / steps if steps else 0.0,
    }


@app.function(image=cpu_image, timeout=1800, volumes=VOLS)
def aggregate(langs: list[str]) -> dict:
    import json
    import os

    out = {"langs": langs, "variants": VARIANTS, "rows": {}}
    for lang in langs:
        out["rows"][lang] = {}
        for variant in VARIANTS:
            path = f"{RESULTS}/{lang}_{variant}.jsonl"
            if not os.path.exists(path):
                continue
            recs = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
            accepted = sum(r["accepted_draft_tokens"] for r in recs)
            proposed = sum(r["proposed_draft_tokens"] for r in recs)
            generated = sum(r["num_generated_tokens"] for r in recs)
            steps = sum(r["forward_steps"] for r in recs)
            seconds = sum(r["spec_seconds"] for r in recs)
            out["rows"][lang][variant] = {
                "n": len(recs),
                "acceptance_rate": round(accepted / proposed, 4) if proposed else 0.0,
                "mean_accept_length": round(generated / steps, 3) if steps else 0.0,
                "tok_s": round(sum(r["emitted_tokens"] for r in recs) / seconds, 2) if seconds else 0.0,
            }

    os.makedirs(RESULTS, exist_ok=True)
    with open(f"{RESULTS}/summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    md = ["# Language full fine-tune comparison\n"]
    md.append("| language | base acc | own LoRA | combined LoRA | full FT |")
    md.append("|---|---:|---:|---:|---:|")
    for lang in langs:
        row = out["rows"].get(lang, {})
        vals = []
        for variant in VARIANTS:
            cell = row.get(variant)
            vals.append(f"{100 * cell['acceptance_rate']:.2f}%" if cell else "-")
        md.append(f"| {lang} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |")
    with open(f"{RESULTS}/report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    data_vol.commit()
    return out


@app.function(image=cpu_image, timeout=24 * 3600, volumes=VOLS)
def full(
    langs: list[str] = DEFAULT_LANGS,
    epochs: int = 3,
    lr: float = 1e-5,
    batch_size: int = 4,
    max_new_tokens: int = 256,
    bench_limit: int = 100,
    skip_train: bool = False,
) -> dict:
    resolved = check_inputs.remote(langs)["langs"]
    if not skip_train:
        train_jobs = {
            lang: train_full.spawn(lang, epochs=epochs, lr=lr, batch_size=batch_size)
            for lang in resolved
        }
        train_logs = {lang: job.get() for lang, job in train_jobs.items()}
        print(train_logs, flush=True)
    bench_jobs = {
        (lang, variant): bench.spawn(lang, variant, max_new_tokens=max_new_tokens,
                                     limit=bench_limit)
        for lang in resolved
        for variant in VARIANTS
    }
    for job in bench_jobs.values():
        job.get()
    return aggregate.remote(resolved)


@app.local_entrypoint()
def smoke():
    langs = ["Polish"]
    check_inputs.remote(langs)
    train_full.remote("Polish", epochs=1, batch_size=1, val_every=1000, max_steps=1)
    for variant in VARIANTS:
        bench.remote("Polish", variant, max_new_tokens=64, limit=3, warmup=1)
    print(aggregate.remote(langs))


@app.local_entrypoint()
def launch(
    langs: str = ",".join(DEFAULT_LANGS),
    epochs: int = 3,
    lr: float = 1e-5,
    batch_size: int = 4,
    max_new_tokens: int = 256,
    bench_limit: int = 100,
    skip_train: bool = False,
):
    parsed_langs = [lang.strip() for lang in langs.split(",") if lang.strip()]
    call = full.spawn(langs=parsed_langs, epochs=epochs, lr=lr, batch_size=batch_size,
                      max_new_tokens=max_new_tokens, bench_limit=bench_limit,
                      skip_train=skip_train)
    print(f"LAUNCHED language full-ft: {call.object_id}")


@app.local_entrypoint()
def results(langs: str = ",".join(DEFAULT_LANGS)):
    parsed_langs = [lang.strip() for lang in langs.split(",") if lang.strip()]
    resolved = check_inputs.remote(parsed_langs)["langs"]
    print(aggregate.remote(resolved))
