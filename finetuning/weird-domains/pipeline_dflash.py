#!/usr/bin/env python3
"""Weird-domains LoRA specialization — DFlash drafter on translation / roleplay / poetry.

The three domains DFlash scored poorly on in the domain benchmarks, and — unlike
the five languages — genuinely heterogeneous TASK TYPES, so the combined-vs-own
comparison here is a real interference test (languages were too similar to
trigger one).

Same one-stack method as ../multilingual/pipeline_langs.py: pretrained
z-lab/Qwen3-8B-DFlash-b16 trained with the vendored SpecForge OnlineDFlashModel
(DFlash matching loss, target's live hidden states, self-distilled data) and
benchmarked with its own spec_generate. Trains FOUR rank-16 LoRAs (one per
domain + one combined) and benches the 3x3 matrix per domain: base vs own vs
combined. The prep (target-generated answers) is SHARED with the EAGLE
weird-domains pipeline for an apples-to-apples cross-speculator comparison.

Run:
    modal run finetuning/weird-domains/pipeline_dflash.py::smoke
    modal run --detach finetuning/weird-domains/pipeline_dflash.py::run
"""
import pathlib

import modal

LOCAL = pathlib.Path(__file__).resolve().parent            # finetuning/weird-domains/
PARENT = LOCAL.parent                                      # finetuning/
LORA = PARENT / "LoRA" / "lora.py"
SPEC_PATCH = PARENT / "spec_patch.py"
ONLINE = PARENT / "online_dflash.py"
DATA = LOCAL / "data"

DRAFT_MODEL = "z-lab/Qwen3-8B-DFlash-b16"
TARGET_MODEL = "Qwen/Qwen3-8B"
GPU = "H200"

DOMAINS = ["translation", "roleplay", "poetry"]
VARIANTS = ["base", "own", "combined"]

app = modal.App("dflash-weird-domains")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.9.1", "transformers==4.57.3", "accelerate>=1.0.0",
                 "datasets>=3.0.0", "numpy")
    .env({"HF_HOME": "/cache", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_file(str(ONLINE), "/root/online_dflash.py")
    .add_local_file(str(LORA), "/root/lora.py")
    .add_local_file(str(SPEC_PATCH), "/root/spec_patch.py")
    .add_local_dir(str(DATA), "/root/data")
)

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .run_commands(
        "pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly",
        "pip install -U transformers hf_transfer",
    )
    .env({
        "HF_HOME": "/cache", "CUDA_HOME": "/usr/local/cuda",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN", "VLLM_USE_FLASHINFER_SAMPLER": "0",
    })
    .add_local_dir(str(DATA), "/root/data")
)

hf_cache = modal.Volume.from_name("dflash-hf-cache", create_if_missing=True)
work = modal.Volume.from_name("code-sql-pipeline", create_if_missing=True)
VOLS = {"/cache": hf_cache, "/work": work}

PREP = "/work/prep/weird"                        # SHARED with the EAGLE pipeline
MODELS = "/work/models/weird_dflash"
RESULTS = "/work/results/weird_dflash"


def _read_prompts(domain, split):
    import json
    out = []
    with open(f"/root/data/{domain}/{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line)["prompt"])
    return out


def _chat_text(tok, prompt):
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def _load_target():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda").eval()
    for p in target.parameters():
        p.requires_grad_(False)
    return tok, target


def _load_draft():
    import torch
    from transformers import AutoModel
    return AutoModel.from_pretrained(
        DRAFT_MODEL, trust_remote_code=True, dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda").eval()


# --------------------------------------------------------------------------- #
# 1) PREP — one vLLM engine, all three domains (train+val); shared volume path
# --------------------------------------------------------------------------- #
@app.function(gpu=GPU, image=vllm_image, timeout=3600, volumes=VOLS)
def prep(max_new_tokens: int = 256, limit: int = 0):
    import os
    import time

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    llm = LLM(model=TARGET_MODEL, dtype="bfloat16", gpu_memory_utilization=0.90,
              max_model_len=2048, enable_prefix_caching=True)
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    summary = {}
    for domain in DOMAINS:
        os.makedirs(f"{PREP}/{domain}", exist_ok=True)
        for split in ("train", "val"):
            prompts = _read_prompts(domain, split)
            if limit:
                prompts = prompts[:limit]
            texts = [_chat_text(tok, p) for p in prompts]
            t0 = time.time()
            outs = llm.generate(texts, sp)
            records = []
            for o in outs:
                pids = list(o.prompt_token_ids)
                gids = list(o.outputs[0].token_ids)
                if len(gids) < 2:
                    continue
                records.append({"input_ids": torch.tensor(pids + gids, dtype=torch.long),
                                "prompt_len": len(pids)})
            torch.save(records, f"{PREP}/{domain}/{split}.pt")
            resp = [len(r["input_ids"]) - r["prompt_len"] for r in records]
            summary[f"{domain}/{split}"] = {
                "n": len(records), "sec": round(time.time() - t0, 1),
                "mean_resp_len": round(sum(resp) / max(len(resp), 1), 1)}
            print(f"[prep:{domain}:{split}] {summary[f'{domain}/{split}']}", flush=True)
    work.commit()
    return summary


# --------------------------------------------------------------------------- #
# 2) TRAIN — one LoRA per domain + one combined (DFlash matching loss)
# --------------------------------------------------------------------------- #
@app.function(gpu=GPU, image=image, timeout=3 * 3600, volumes=VOLS)
def train_lora(name: str, domains: list, epochs: int = 3, lr: float = 1e-3,
               batch_size: int = 12, num_anchors: int = 48, max_seq_len: int = 512,
               val_every: int = 200):
    import os
    import sys
    import time

    import torch

    sys.path.insert(0, "/root")
    from lora import inject_lora, lora_state_dict, lora_trainable_parameters
    from online_dflash import OnlineDFlashModel

    tok, target = _load_target()
    draft = _load_draft()
    extract = sys.modules[type(draft).__module__].extract_context_feature
    target_layer_ids = draft.target_layer_ids
    mask_token_id = draft.mask_token_id
    block_size = draft.block_size

    replaced = inject_lora(draft, rank=16, alpha=32,
                           target_modules=("q_proj", "k_proj", "v_proj", "o_proj"))
    draft.to("cuda", dtype=torch.bfloat16)
    trainable = list(lora_trainable_parameters(draft))
    print(f"[train:{name}] {len(replaced)} layers, "
          f"{sum(p.numel() for p in trainable):,} params, domains={domains}", flush=True)

    online = OnlineDFlashModel(
        draft_model=draft, target_lm_head=target.lm_head,
        target_embed_tokens=target.get_input_embeddings(),
        mask_token_id=mask_token_id, block_size=block_size,
        attention_backend="sdpa", num_anchors=num_anchors,
        loss_decay_gamma=7.0, loss_type="dflash",
    )
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)

    train_data, val_data = [], []
    for d in domains:
        train_data += torch.load(f"{PREP}/{d}/train.pt")
        val_data += torch.load(f"{PREP}/{d}/val.pt")
    print(f"[train:{name}] {len(train_data)} train / {len(val_data)} val", flush=True)

    def make_batch(records):
        seqs = [r["input_ids"][:max_seq_len] for r in records]
        S = max(len(s) for s in seqs)
        B = len(seqs)
        input_ids = torch.full((B, S), tok.pad_token_id, dtype=torch.long)
        loss_mask = torch.zeros((B, S), dtype=torch.float)
        for i, (s, r) in enumerate(zip(seqs, records)):
            L = len(s)
            input_ids[i, :L] = s
            loss_mask[i, min(r["prompt_len"], L):L] = 1.0
        return input_ids.to("cuda"), loss_mask.to("cuda")

    @torch.no_grad()
    def target_hidden(input_ids):
        pos = torch.arange(input_ids.shape[1], device="cuda").unsqueeze(0).expand(input_ids.shape[0], -1)
        out = target(input_ids=input_ids, position_ids=pos, use_cache=False, output_hidden_states=True)
        return extract(out.hidden_states, target_layer_ids).clone()

    def run_batch(records, train=True):
        input_ids, loss_mask = make_batch(records)
        anchorable = loss_mask[:, : max(loss_mask.shape[1] - block_size, 0) + 1].sum(dim=1)
        keep = anchorable > block_size + 1
        if keep.sum() == 0:
            return None
        input_ids, loss_mask = input_ids[keep], loss_mask[keep]
        hid = target_hidden(input_ids)
        try:
            loss, acc, _ = online(input_ids=input_ids, hidden_states=hid, loss_mask=loss_mask)
        except ValueError:
            return None
        if train:
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
        return float(loss.item()), float(acc.item())

    @torch.no_grad()
    def validate():
        online.eval()
        ls, accs = [], []
        stride = max(len(val_data) // 60, 1)
        sample = val_data[::stride][:60]
        for i in range(0, len(sample), batch_size):
            r = run_batch(sample[i:i + batch_size], train=False)
            if r:
                ls.append(r[0]); accs.append(r[1])
        online.train()
        return (sum(ls) / len(ls), sum(accs) / len(accs)) if ls else (None, None)

    log = {"name": name, "domains": domains, "epochs": epochs, "lr": lr}
    online.train()
    step, t0 = 0, time.time()
    log["val_initial"] = validate()
    print(f"[train:{name}] initial val {log['val_initial']}", flush=True)
    for ep in range(epochs):
        order = torch.randperm(len(train_data)).tolist()
        for i in range(0, len(train_data), batch_size):
            r = run_batch([train_data[j] for j in order[i:i + batch_size]], train=True)
            if r is None:
                continue
            step += 1
            if step % 50 == 0:
                print(f"[train:{name}] ep{ep} step{step} loss={r[0]:.4f} acc={r[1]:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if step % val_every == 0:
                vl, va = validate()
                print(f"[train:{name}]   val step{step} loss={vl:.4f} acc={va:.4f}", flush=True)
    log["val_final"] = validate()
    print(f"[train:{name}] final val {log['val_final']}", flush=True)

    os.makedirs(MODELS, exist_ok=True)
    path = f"{MODELS}/{name}_lora.pt"
    torch.save(lora_state_dict(draft), path)
    work.commit()
    log["saved"] = path
    log["train_seconds"] = round(time.time() - t0, 1)
    print(f"[train:{name}] saved -> {path}", flush=True)
    return log


# --------------------------------------------------------------------------- #
# 3) BENCH — base vs own vs combined per domain (real spec_generate)
# --------------------------------------------------------------------------- #
def _load_lora_into(draft, sd):
    import torch
    from lora import LoRALinear
    named = dict(draft.named_modules())
    for name, entry in sd.items():
        m = named.get(name)
        assert isinstance(m, LoRALinear), f"missing LoRALinear at {name}"
        with torch.no_grad():
            m.A.copy_(entry["A"].to(m.A.device, m.A.dtype))
            m.B.copy_(entry["B"].to(m.B.device, m.B.dtype))
        m.scaling = float(entry["scaling"])


@app.function(gpu=GPU, image=image, timeout=3 * 3600, volumes=VOLS)
def bench(domain: str, variant: str, max_new_tokens: int = 256, limit: int = 100,
          warmup: int = 2):
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

    if variant != "base":
        from lora import inject_lora
        inject_lora(draft, rank=16, alpha=32,
                    target_modules=("q_proj", "k_proj", "v_proj", "o_proj"))
        draft.to("cuda", dtype=torch.bfloat16)
        adapter = f"{MODELS}/{domain if variant == 'own' else 'combined'}_lora.pt"
        _load_lora_into(draft, torch.load(adapter, map_location="cuda"))
        print(f"[bench:{domain}:{variant}] loaded {adapter}", flush=True)

    draft.spec_generate = make_instrumented_spec_generate(draft)
    block_size = draft.block_size
    stop_ids = [tok.eos_token_id]
    prompts = _read_prompts(domain, "test")[:limit]

    def build_ids(p):
        return tok([_chat_text(tok, p)], return_tensors="pt").input_ids.to("cuda")

    def first_stop(ids, s):
        for i, t in enumerate(ids):
            if t in s:
                return ids[:i + 1]
        return ids

    warm = build_ids(prompts[0])
    for _ in range(warmup):
        with torch.inference_mode():
            draft.spec_generate(target=target, input_ids=warm, max_new_tokens=64,
                                stop_token_ids=stop_ids, temperature=0.0)
    torch.cuda.synchronize()

    os.makedirs(RESULTS, exist_ok=True)
    out_path = f"{RESULTS}/{domain}_{variant}.jsonl"
    recs = []
    t_start = time.time()
    with open(out_path, "w") as fout:
        for idx, prompt in enumerate(prompts):
            input_ids = build_ids(prompt)
            n_in = input_ids.shape[1]
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.inference_mode():
                out_ids, committed = draft.spec_generate(
                    target=target, input_ids=input_ids, max_new_tokens=max_new_tokens,
                    stop_token_ids=stop_ids, temperature=0.0)
            torch.cuda.synchronize(); spec_dt = time.perf_counter() - t0
            spec_gen = first_stop(out_ids[0, n_in:].tolist(), stop_ids)
            spec_tok_s = len(spec_gen) / spec_dt if spec_dt > 0 else 0.0
            steps = len(committed)
            gen = sum(committed)
            accepted = sum(c - 1 for c in committed)
            proposed = steps * (block_size - 1)
            rec = {"category": f"{domain}:{variant}", "lang": domain, "variant": variant,
                   "prompt_idx": idx, "num_input_tokens": n_in,
                   "forward_steps": steps, "num_generated_tokens": gen,
                   "accepted_draft_tokens": accepted, "proposed_draft_tokens": proposed,
                   "acceptance_rate": (accepted / proposed) if proposed else 0.0,
                   "mean_accept_length": (gen / steps) if steps else 0.0,
                   "spec_seconds": spec_dt, "spec_tok_s": spec_tok_s}
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            recs.append(rec)
            if idx % 20 == 0:
                print(f"[bench:{domain}:{variant}] {idx+1}/{len(prompts)} "
                      f"accept={rec['acceptance_rate']*100:.0f}% "
                      f"mean_len={rec['mean_accept_length']:.2f} "
                      f"({time.time()-t_start:.0f}s)", flush=True)
    work.commit()

    n = len(recs)
    tp = sum(r["proposed_draft_tokens"] for r in recs)
    summ = {"domain": domain, "variant": variant, "n": n,
            "acceptance_rate_pooled": (sum(r["accepted_draft_tokens"] for r in recs) / tp) if tp else 0.0,
            "mean_accept_length": sum(r["mean_accept_length"] for r in recs) / n}
    print(f"[bench:{domain}:{variant}] DONE {summ}", flush=True)
    return summ


# --------------------------------------------------------------------------- #
# aggregate — 3x3 matrix
# --------------------------------------------------------------------------- #
@app.function(image=image, timeout=1200, volumes=VOLS)
def aggregate():
    import json

    work.reload()
    rows = {}
    merged = f"{RESULTS}/weird_dflash.jsonl"
    with open(merged, "w") as fout:
        for domain in DOMAINS:
            for variant in VARIANTS:
                try:
                    recs = []
                    with open(f"{RESULTS}/{domain}_{variant}.jsonl") as f:
                        for line in f:
                            if line.strip():
                                recs.append(json.loads(line)); fout.write(line)
                except FileNotFoundError:
                    continue
                if not recs:
                    continue
                n = len(recs)
                tp = sum(r["proposed_draft_tokens"] for r in recs)
                rows[(domain, variant)] = {
                    "n": n,
                    "accept": (sum(r["accepted_draft_tokens"] for r in recs) / tp) if tp else 0.0,
                    "mean_len": sum(r["mean_accept_length"] for r in recs) / n,
                }

    csv = ["domain,variant,n,acceptance_rate_pooled,mean_accept_length"]
    for (domain, variant), d in rows.items():
        csv.append(f"{domain},{variant},{d['n']},{d['accept']:.4f},{d['mean_len']:.4f}")
    csv_text = "\n".join(csv) + "\n"

    md = ["# Weird-domains LoRA specialization (DFlash) — base vs own vs combined\n",
          "Domains: translation · roleplay · poetry (heterogeneous task types — a real "
          "interference test for the combined adapter). Target **Qwen/Qwen3-8B** · "
          "drafter **z-lab/Qwen3-8B-DFlash-b16** · temperature 0 · rank-16 LoRA.\n",
          "| domain | base accept | own accept | combined accept | base len | own len | combined len |",
          "|---|--:|--:|--:|--:|--:|--:|"]
    wins_own, wins_comb = 0, 0
    for domain in DOMAINS:
        b = rows.get((domain, "base")); o = rows.get((domain, "own")); c = rows.get((domain, "combined"))
        if not b:
            continue
        def acc(d, ref=None):
            if d is None:
                return "—"
            s = f"{d['accept']*100:.1f}%"
            if ref is not None:
                diff = (d["accept"] - ref["accept"]) * 100
                s += f" ({'+' if diff >= 0 else ''}{diff:.1f}pp)"
            return s
        def ml(d, ref=None):
            if d is None:
                return "—"
            s = f"{d['mean_len']:.2f}"
            if ref is not None:
                diff = d["mean_len"] - ref["mean_len"]
                s += f" ({'+' if diff >= 0 else ''}{diff:.2f})"
            return s
        if o and o["accept"] > b["accept"]:
            wins_own += 1
        if c and c["accept"] > b["accept"]:
            wins_comb += 1
        md.append(f"| {domain} | {acc(b)} | {acc(o, b)} | {acc(c, b)} "
                  f"| {ml(b)} | {ml(o, b)} | {ml(c, b)} |")
    md.append(f"\n**own-LoRA beats base on {wins_own}/{len(DOMAINS)} domains; "
              f"combined beats base on {wins_comb}/{len(DOMAINS)}.** own-vs-combined "
              "gaps indicate cross-task interference inside one adapter.\n")
    md_text = "\n".join(md)

    with open(f"{RESULTS}/weird_dflash_report.md", "w") as f:
        f.write(md_text)
    with open(f"{RESULTS}/weird_dflash_comparison.csv", "w") as f:
        f.write(csv_text)
    work.commit()
    print("\n" + md_text)
    return {"_report.md": md_text, "_comparison.csv": csv_text}


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
@app.function(image=image, timeout=10 * 3600, volumes=VOLS)
def orchestrate(epochs: int = 3, bench_limit: int = 100, skip_prep: bool = False,
                prep_limit: int = 0):
    out = {}
    if not skip_prep:
        out["prep"] = prep.remote(limit=prep_limit)
        print("[orchestrate] prep done", flush=True)

    jobs = {d: train_lora.spawn(d, [d], epochs=epochs) for d in DOMAINS}
    jobs["combined"] = train_lora.spawn("combined", DOMAINS, epochs=epochs)
    out["train"] = {name: c.get().get("val_final") for name, c in jobs.items()}
    print(f"[orchestrate] train done: {out['train']}", flush=True)

    bjobs = {(d, v): bench.spawn(d, v, limit=bench_limit)
             for d in DOMAINS for v in VARIANTS}
    out["bench"] = {f"{d}:{v}": c.get() for (d, v), c in bjobs.items()}
    print("[orchestrate] bench done", flush=True)

    agg = aggregate.remote()
    out["report"] = agg["_report.md"]
    print("\n" + agg["_report.md"], flush=True)
    return out


@app.local_entrypoint()
def run(epochs: int = 3, bench_limit: int = 100, skip_prep: bool = False):
    out = orchestrate.remote(epochs=epochs, bench_limit=bench_limit, skip_prep=skip_prep)
    print(out.get("report", out))


@app.local_entrypoint()
def launch(epochs: int = 3, bench_limit: int = 100, skip_prep: bool = False):
    """Fire-and-forget: spawn the server-side orchestrator and exit immediately
    (use with --detach on a flaky network; client needed only ~2s)."""
    call = orchestrate.spawn(epochs=epochs, bench_limit=bench_limit, skip_prep=skip_prep)
    print(f"LAUNCHED orchestrate: {call.object_id}")


@app.function(image=image, timeout=6 * 3600, volumes=VOLS)
def bench_all(bench_limit: int = 100):
    """Re-run ONLY the 9 benches (adapters already trained) + aggregate."""
    bjobs = {(d, v): bench.spawn(d, v, limit=bench_limit)
             for d in DOMAINS for v in VARIANTS}
    out = {f"{d}:{v}": c.get() for (d, v), c in bjobs.items()}
    agg = aggregate.remote()
    print("\n" + agg["_report.md"], flush=True)
    return {"bench": out, "report": agg["_report.md"]}


@app.local_entrypoint()
def launch_bench(bench_limit: int = 100):
    call = bench_all.spawn(bench_limit=bench_limit)
    print(f"LAUNCHED bench_all: {call.object_id}")


@app.local_entrypoint()
def agg_only():
    print(aggregate.remote()["_report.md"])


@app.local_entrypoint()
def smoke():
    """Tiny validation (machinery is already validated on multilingual; this just
    checks the new data paths + volume prefixes): prep 8/domain, 1-epoch
    translation train, 2-prompt bench x3, aggregate."""
    import json

    print("=== SMOKE prep (8 prompts/domain) ===")
    print(json.dumps(prep.remote(max_new_tokens=96, limit=8), indent=2))
    print("=== SMOKE train translation + combined (1 epoch) ===")
    tj = train_lora.spawn("translation", ["translation"], epochs=1, val_every=1000)
    cj = train_lora.spawn("combined", DOMAINS, epochs=1, val_every=1000)
    print("[translation]", tj.get().get("saved"), " [combined]", cj.get().get("saved"))
    print("=== SMOKE bench translation x3 (2 prompts) ===")
    bj = {v: bench.spawn("translation", v, max_new_tokens=64, limit=2, warmup=1)
          for v in VARIANTS}
    for v, c in bj.items():
        print(v, json.dumps(c.get(), indent=2))
    print("=== SMOKE aggregate ===")
    print(aggregate.remote()["_report.md"])
