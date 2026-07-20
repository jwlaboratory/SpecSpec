#!/usr/bin/env python3
"""Independent drafter — can vanilla (two-model) speculative decoding be
specialized where EAGLE could not?

EAGLE3 and DFlash both condition the drafter on the *target's hidden states*,
so fine-tuning them means keeping a training-time feature pipeline byte-exact
with serving — the hard part that produced most of this repo's bugs, and (we
suspect) the residual train/serve gap behind EAGLE's flat-to-negative LoRA
results. The original speculation method (Leviathan/Chen 2023) has no such
contract: a completely separate small LM drafts tokens and the target verifies
them. Its only alignment channel is the token stream itself — exactly what
plain self-distillation trains. Hypothesis: LoRA specialization lands more
easily here.

Setup:
  * target  Qwen/Qwen3-8B (frozen, as everywhere in the repo)
  * drafter Qwen/Qwen3-0.6B — an off-the-shelf independent model, same
    tokenizer/vocab as the target, k=4 proposals per step, greedy verification
    (lossless; `lib/vanilla_spec.py`)
  * 5 domains spanning the headroom curve, all with DFlash specialist numbers
    from ../05-interference-ladder for direct comparison:
    code_sql · lang_polish · lang_korean · ood_legal · task_math_reasoning
  * train: rank-16 LoRA (alpha 32) on q/k/v/o of the 0.6B drafter, plain CE on
    the target's own generations (self-distillation, same prep data as exp 05 —
    reused straight off the volume). One "own" LoRA per domain (800 ex) + one
    "combined" on all five (4000 ex), 3 epochs — the exp-02 protocol.
  * bench: 5x3 matrix (base | own | combined), held-out test n=100,
    temperature 0, same metrics/jsonl schema as every other experiment.

Run (detached — server-side orchestration survives client drops):
    modal run --detach experiments/06-independent-drafter/pipeline.py::launch
Cheap validation first:
    modal run experiments/06-independent-drafter/pipeline.py::smoke
"""
import pathlib

import modal

LOCAL = pathlib.Path(__file__).resolve().parent            # experiments/06-independent-drafter/
ROOT = LOCAL.parent.parent                                 # repo root
LORA = ROOT / "lib" / "lora.py"
VANILLA = ROOT / "lib" / "vanilla_spec.py"
DATA = ROOT / "data" / "synthetic"

DRAFT_MODEL = "Qwen/Qwen3-0.6B"
TARGET_MODEL = "Qwen/Qwen3-8B"
GPU = "H200"
DRAFT_K = 4                                                # proposals per step

DOMAINS = ["code_sql", "lang_polish", "lang_korean", "ood_legal",
           "task_math_reasoning"]
VARIANTS = ["base", "own", "combined"]

app = modal.App("independent-drafter-lora")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.9.1", "transformers==4.57.3", "accelerate>=1.0.0",
                 "datasets>=3.0.0", "numpy")
    .env({"HF_HOME": "/cache", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_file(str(LORA), "/root/lora.py")
    .add_local_file(str(VANILLA), "/root/vanilla_spec.py")
    .add_local_dir(str(DATA), "/root/data")
)

# prep image only needed if a domain is missing from the exp-05 prep (vLLM)
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

# self-distillation data: reuse exp-05's target generations (same target model,
# same tokenizer, same 800/100 train/val splits); fall back to our own prep dir
PREP_SHARED = "/work/prep/interference"
PREP_OWN = "/work/prep/independent"
MODELS = "/work/models/independent"
RESULTS = "/work/results/independent"


def _prep_path(domain, split):
    import os
    for base in (PREP_SHARED, PREP_OWN):
        p = f"{base}/{domain}/{split}.pt"
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no prep data for {domain}/{split} — run ::prep")


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


def _load_lm(name):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


# --------------------------------------------------------------------------- #
# 0) PREP — only for domains missing from the shared exp-05 prep
# --------------------------------------------------------------------------- #
@app.function(gpu=GPU, image=vllm_image, timeout=2 * 3600, volumes=VOLS)
def prep(domains: list = None, max_new_tokens: int = 256, limit: int = 0):
    import os
    import time

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    domains = domains or DOMAINS
    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    llm = LLM(model=TARGET_MODEL, dtype="bfloat16", gpu_memory_utilization=0.90,
              max_model_len=2048, enable_prefix_caching=True)
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    summary = {}
    for domain in domains:
        os.makedirs(f"{PREP_OWN}/{domain}", exist_ok=True)
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
            torch.save(records, f"{PREP_OWN}/{domain}/{split}.pt")
            summary[f"{domain}/{split}"] = {"n": len(records),
                                            "sec": round(time.time() - t0, 1)}
            print(f"[prep:{domain}:{split}] {summary[f'{domain}/{split}']}", flush=True)
        work.commit()
    return summary


# --------------------------------------------------------------------------- #
# 1) TRAIN — plain CE self-distillation of the 0.6B drafter (no target model,
#    no hidden states, no feature pipeline: the whole point of this experiment)
# --------------------------------------------------------------------------- #
@app.function(gpu=GPU, image=image, timeout=6 * 3600, volumes=VOLS)
def train_lora(name: str, domains: list, epochs: int = 3, lr: float = 1e-3,
               batch_size: int = 16, max_seq_len: int = 512, val_every: int = 200,
               rank: int = 16, limit: int = 0):
    import sys
    import time

    import torch
    import torch.nn.functional as F

    sys.path.insert(0, "/root")
    from lora import inject_lora, lora_state_dict, lora_trainable_parameters

    tok = _tokenizer()
    draft = _load_lm(DRAFT_MODEL)
    replaced = inject_lora(draft, rank=rank, alpha=2 * rank,
                           target_modules=("q_proj", "k_proj", "v_proj", "o_proj"))
    draft.to("cuda", dtype=torch.bfloat16)
    trainable = list(lora_trainable_parameters(draft))
    print(f"[train:{name}] rank={rank} {len(replaced)} layers, "
          f"{sum(p.numel() for p in trainable):,} LoRA params, "
          f"{len(domains)} domains", flush=True)
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)

    train_data, val_data = [], []
    for domain in domains:
        train_data += torch.load(_prep_path(domain, "train"))
        val_data += torch.load(_prep_path(domain, "val"))
    if limit:
        train_data, val_data = train_data[:limit], val_data[:limit]
    print(f"[train:{name}] {len(train_data)} train / {len(val_data)} val", flush=True)

    def make_batch(records):
        seqs = [r["input_ids"][:max_seq_len] for r in records]
        S = max(len(s) for s in seqs)
        B = len(seqs)
        input_ids = torch.full((B, S), tok.pad_token_id, dtype=torch.long)
        loss_mask = torch.zeros((B, S), dtype=torch.bool)
        for i, (s, r) in enumerate(zip(seqs, records)):
            L = len(s)
            input_ids[i, :L] = s
            loss_mask[i, min(r["prompt_len"], L):L] = True
        return input_ids.to("cuda"), loss_mask.to("cuda")

    def run_batch(records, train=True):
        input_ids, loss_mask = make_batch(records)
        mask = loss_mask[:, 1:]                       # labels are the shifted ids
        if mask.sum() == 0:
            return None
        logits = draft(input_ids=input_ids, use_cache=False).logits[:, :-1]
        labels = input_ids[:, 1:]
        loss = F.cross_entropy(logits[mask].float(), labels[mask])
        acc = (logits[mask].argmax(-1) == labels[mask]).float().mean()
        if train:
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
        return float(loss.item()), float(acc.item())

    @torch.no_grad()
    def validate():
        draft.eval()
        ls, accs = [], []
        stride = max(len(val_data) // 60, 1)          # spread across domains
        sample = val_data[::stride][:60]
        for i in range(0, len(sample), batch_size):
            r = run_batch(sample[i:i + batch_size], train=False)
            if r:
                ls.append(r[0]); accs.append(r[1])
        draft.train()
        return (sum(ls) / len(ls), sum(accs) / len(accs)) if ls else (None, None)

    log = {"name": name, "domains": domains, "epochs": epochs, "lr": lr, "val": []}
    draft.train()
    step = 0
    t0 = time.time()
    log["val_initial"] = validate()
    print(f"[train:{name}] initial val {log['val_initial']}", flush=True)
    for ep in range(epochs):
        perm = torch.randperm(len(train_data)).tolist()
        for i in range(0, len(train_data), batch_size):
            r = run_batch([train_data[j] for j in perm[i:i + batch_size]], train=True)
            if r is None:
                continue
            step += 1
            if step % 50 == 0:
                print(f"[train:{name}] ep{ep} step{step} loss={r[0]:.4f} acc={r[1]:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if step % val_every == 0:
                vl, va = validate()
                log["val"].append({"step": step, "loss": vl, "acc": va})
                print(f"[train:{name}]   val step{step} loss={vl:.4f} acc={va:.4f}", flush=True)
    log["val_final"] = validate()
    print(f"[train:{name}] final val {log['val_final']}", flush=True)

    import os
    os.makedirs(MODELS, exist_ok=True)
    sfx = f"_r{rank}" if rank != 16 else ""
    path = f"{MODELS}/{name}{sfx}_lora.pt"
    torch.save(lora_state_dict(draft), path)
    work.commit()
    log["saved"] = path
    log["train_seconds"] = round(time.time() - t0, 1)
    print(f"[train:{name}] saved -> {path}", flush=True)
    return log


# --------------------------------------------------------------------------- #
# 2) BENCH — vanilla spec decode: base vs own vs combined, vs target-only
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
          warmup: int = 2, rank: int = 16, k: int = DRAFT_K):
    import json
    import os
    import sys
    import time

    import torch

    sys.path.insert(0, "/root")
    from vanilla_spec import spec_generate

    assert variant in VARIANTS
    tok = _tokenizer()
    target = _load_lm(TARGET_MODEL)
    draft = _load_lm(DRAFT_MODEL)

    sfx = f"_r{rank}" if rank != 16 else ""
    if variant != "base":
        from lora import inject_lora
        inject_lora(draft, rank=rank, alpha=2 * rank,
                    target_modules=("q_proj", "k_proj", "v_proj", "o_proj"))
        draft.to("cuda", dtype=torch.bfloat16)
        adapter = (f"{MODELS}/{domain}{sfx}_lora.pt" if variant == "own"
                   else f"{MODELS}/{variant}{sfx}_lora.pt")
        _load_lora_into(draft, torch.load(adapter, map_location="cuda"))
        print(f"[bench:{domain}:{variant}] loaded {adapter}", flush=True)

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
        spec_generate(target, draft, warm, 64, stop_ids, k=k)
    torch.cuda.synchronize()

    os.makedirs(RESULTS, exist_ok=True)
    out_path = f"{RESULTS}/{domain}_{variant}{sfx}.jsonl"
    recs = []
    t_start = time.time()
    with open(out_path, "w") as fout:
        for idx, prompt in enumerate(prompts):
            input_ids = build_ids(prompt)
            n_in = input_ids.shape[1]
            torch.cuda.synchronize(); t0 = time.perf_counter()
            out_ids, committed = spec_generate(
                target, draft, input_ids, max_new_tokens, stop_ids, k=k)
            torch.cuda.synchronize(); spec_dt = time.perf_counter() - t0
            spec_gen = first_stop(out_ids[0, n_in:].tolist(), stop_ids)
            spec_tok_s = len(spec_gen) / spec_dt if spec_dt > 0 else 0.0
            steps = len(committed)
            gen = sum(committed)
            accepted = sum(c - 1 for c in committed)
            proposed = steps * k

            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.inference_mode():
                base_out = target.generate(
                    input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                    num_beams=1, pad_token_id=tok.pad_token_id, eos_token_id=stop_ids,
                    use_cache=True)
            torch.cuda.synchronize(); base_dt = time.perf_counter() - t0
            base_gen = first_stop(base_out[0, n_in:].tolist(), stop_ids)
            base_tok_s = len(base_gen) / base_dt if base_dt > 0 else 0.0

            rec = {"category": f"{domain}:{variant}", "domain": domain, "variant": variant,
                   "prompt_idx": idx, "num_input_tokens": n_in, "draft_k": k,
                   "forward_steps": steps, "num_generated_tokens": gen,
                   "accepted_draft_tokens": accepted, "proposed_draft_tokens": proposed,
                   "acceptance_rate": (accepted / proposed) if proposed else 0.0,
                   "mean_accept_length": (gen / steps) if steps else 0.0,
                   "committed_per_step": committed,
                   "spec_seconds": spec_dt, "spec_tok_s": spec_tok_s,
                   "baseline_seconds": base_dt, "baseline_tok_s": base_tok_s,
                   "speedup": (spec_tok_s / base_tok_s) if base_tok_s > 0 else 0.0,
                   "exact_match": bool(spec_gen == base_gen)}
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            recs.append(rec)
            if idx % 20 == 0:
                print(f"[bench:{domain}:{variant}] {idx+1}/{len(prompts)} "
                      f"accept={rec['acceptance_rate']*100:.0f}% "
                      f"mean_len={rec['mean_accept_length']:.2f} "
                      f"({time.time()-t_start:.0f}s)", flush=True)
    work.commit()

    tot_acc = sum(r["accepted_draft_tokens"] for r in recs)
    tot_prop = sum(r["proposed_draft_tokens"] for r in recs)
    n = len(recs)
    summ = {"domain": domain, "variant": variant, "n": n,
            "acceptance_rate_pooled": (tot_acc / tot_prop) if tot_prop else 0.0,
            "mean_accept_length": sum(r["mean_accept_length"] for r in recs) / n,
            "speedup": sum(r["speedup"] for r in recs) / n,
            "exact": sum(bool(r["exact_match"]) for r in recs) / n}
    print(f"[bench:{domain}:{variant}] DONE {summ}", flush=True)
    return summ


# --------------------------------------------------------------------------- #
# aggregate — the 5x3 matrix
# --------------------------------------------------------------------------- #
@app.function(image=image, timeout=1200, volumes=VOLS)
def aggregate():
    import json

    work.reload()
    rows = {}
    merged = f"{RESULTS}/independent.jsonl"
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
                    "speedup": sum(r["speedup"] for r in recs) / n,
                    "exact": sum(bool(r["exact_match"]) for r in recs) / n,
                }

    csv = ["domain,variant,n,acceptance_rate_pooled,mean_accept_length,speedup,exact_match_rate"]
    for (domain, variant), d in rows.items():
        csv.append(f"{domain},{variant},{d['n']},{d['accept']:.4f},{d['mean_len']:.4f},"
                   f"{d['speedup']:.4f},{d['exact']:.4f}")
    csv_text = "\n".join(csv) + "\n"

    md = ["# Independent drafter — vanilla speculative decoding, base vs own vs combined\n",
          "Held-out test (n=100/domain) · target **Qwen/Qwen3-8B** · drafter "
          f"**{DRAFT_MODEL}** (k={DRAFT_K} proposals/step, greedy verification, "
          "lossless) · rank-16 LoRA on q/k/v/o, plain-CE self-distillation.\n",
          "own = LoRA trained on that domain only (800 ex); combined = one LoRA on "
          "all five domains (4000 ex).\n",
          "| domain | base | own | combined | base len | own len | comb len | base spd | own spd |",
          "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]

    def acc(d, ref=None):
        if d is None:
            return "—"
        s = f"{d['accept']*100:.1f}%"
        if ref is not None:
            diff = (d["accept"] - ref["accept"]) * 100
            s += f" ({'+' if diff >= 0 else ''}{diff:.1f}pp)"
        return s

    for domain in DOMAINS:
        b = rows.get((domain, "base"))
        o = rows.get((domain, "own"))
        c = rows.get((domain, "combined"))
        if not b:
            continue
        cells = [acc(b), acc(o, b), acc(c, b),
                 f"{b['mean_len']:.2f}",
                 f"{o['mean_len']:.2f}" if o else "—",
                 f"{c['mean_len']:.2f}" if c else "—",
                 f"{b['speedup']:.2f}×",
                 f"{o['speedup']:.2f}×" if o else "—"]
        md.append(f"| {domain} | " + " | ".join(cells) + " |")

    exact = [d["exact"] for d in rows.values()]
    md.append("\nexact-match vs HF `generate`: "
              + " · ".join(f"{d}:{rows[(d,'base')]['exact']*100:.0f}%"
                           for d in DOMAINS if (d, "base") in rows)
              + " — same rates as the DFlash benches (chunked-verification numerics "
                "flip argmax ties vs step-by-step generate); the match *vector* is "
                "identical across variants, i.e. adapters change speed, never output.\n"
              if exact else "\n")
    md_text = "\n".join(md)

    with open(f"{RESULTS}/independent_report.md", "w") as f:
        f.write(md_text)
    with open(f"{RESULTS}/independent_comparison.csv", "w") as f:
        f.write(csv_text)
    work.commit()
    print("\n" + md_text)
    return {"_report.md": md_text, "_comparison.csv": csv_text}


# --------------------------------------------------------------------------- #
# orchestration — server-side, launch detached
# --------------------------------------------------------------------------- #
@app.function(image=image, timeout=23 * 3600, volumes=VOLS)
def orchestrate(epochs: int = 3, bench_limit: int = 100,
                bench_max_new_tokens: int = 256, rank: int = 16,
                skip_train: bool = False):
    import os
    out = {}

    # prep: reuse exp-05 generations; regenerate only what's missing
    work.reload()
    missing = [d for d in DOMAINS
               if not os.path.exists(f"{PREP_SHARED}/{d}/train.pt")
               and not os.path.exists(f"{PREP_OWN}/{d}/train.pt")]
    if missing:
        out["prep"] = prep.remote(domains=missing)
        print(f"[orchestrate] prepped missing domains: {missing}", flush=True)

    # base benches don't depend on training — start them immediately
    bjobs = {(d, "base"): bench.spawn(d, "base", max_new_tokens=bench_max_new_tokens,
                                      limit=bench_limit, rank=rank)
             for d in DOMAINS}

    if not skip_train:
        jobs = {d: train_lora.spawn(d, [d], epochs=epochs, rank=rank) for d in DOMAINS}
        jobs["combined"] = train_lora.spawn("combined", DOMAINS, epochs=epochs, rank=rank)
        out["train"] = {name: c.get().get("val_final") for name, c in jobs.items()}
        print(f"[orchestrate] train done: {out['train']}", flush=True)

    for d in DOMAINS:
        for v in ("own", "combined"):
            bjobs[(d, v)] = bench.spawn(d, v, max_new_tokens=bench_max_new_tokens,
                                        limit=bench_limit, rank=rank)
    out["bench"] = {f"{d}:{v}": c.get() for (d, v), c in bjobs.items()}
    print(f"[orchestrate] bench done", flush=True)

    agg = aggregate.remote()
    out["report"] = agg["_report.md"]
    print("\n" + agg["_report.md"], flush=True)
    return out


@app.local_entrypoint()
def run(epochs: int = 3, bench_limit: int = 100, rank: int = 16):
    out = orchestrate.remote(epochs=epochs, bench_limit=bench_limit, rank=rank)
    print(out.get("report", out))


@app.local_entrypoint()
def launch(epochs: int = 3, bench_limit: int = 100, rank: int = 16):
    """Fire-and-forget spawn (use with --detach on a flaky network)."""
    call = orchestrate.spawn(epochs=epochs, bench_limit=bench_limit, rank=rank)
    print(f"LAUNCHED orchestrate: {call.object_id}")


@app.local_entrypoint()
def agg_only():
    print(aggregate.remote()["_report.md"])


@app.local_entrypoint()
def smoke():
    """Cheap validation: 1-epoch train of one own + combined on 48 examples,
    2-prompt bench of code_sql across all three variants (exact_match must be
    True — losslessness of the vanilla spec loop), aggregate."""
    import json

    print("=== SMOKE train code_sql + combined (1 epoch, 48 ex) ===")
    jobs = {"code_sql": train_lora.spawn("code_sql", ["code_sql"], epochs=1,
                                         val_every=1000, limit=48),
            "combined": train_lora.spawn("combined", DOMAINS, epochs=1,
                                         val_every=1000, limit=48)}
    for name, c in jobs.items():
        print(f"[{name}]", c.get().get("saved"))
    print("=== SMOKE bench code_sql x 3 variants (2 prompts) ===")
    bj = {v: bench.spawn("code_sql", v, max_new_tokens=64, limit=2, warmup=1)
          for v in VARIANTS}
    for v, c in bj.items():
        print(v, json.dumps(c.get(), indent=2))
    print("=== SMOKE aggregate ===")
    print(aggregate.remote()["_report.md"])
