#!/usr/bin/env python3
"""Exp2-speedup: wall-clock cost of DYNAMICALLY SWITCHING LoRA adapters on a
mixed-language request stream.

Exp1 benched each language in isolation (one container per language x variant),
which hides the serving question: if requests arrive in mixed language order,
the per-language ("own") adapters must be hot-swapped between requests, while
the combined adapter -- and the base drafter -- never swap. This experiment
measures that directly:

  1. build_testset (CPU): from the exp1 volume, keep every language whose
     fetched TRAIN split has >= 1000 prompts AND has a trained own-LoRA;
     take n_per_lang held-out TEST prompts each and shuffle them into ONE
     mixed stream (fixed seed -> consecutive prompts almost always change
     language, the worst case for adapter switching).
  2. bench_stream (H200): ONE container per variant runs the whole stream
     back-to-back through the instrumented spec_generate:
       - base:     no adapter, no swaps
       - own:      per-language adapter; whenever the language changes, copy
                   that language's A/B/scaling into the injected LoRALinears
                   (adapters preloaded in CPU RAM, i.e. a realistic hot-swap).
                   Swap time is measured per switch and COUNTS toward the
                   stream wall-clock.
       - combined: the single all-language adapter loaded once
     Vanilla (no-spec) decode is paired IN-CONTAINER on a subset of the base
     run (memory: wallclock-analytic-model -- never compare across containers).
  3. aggregate: per variant -- stream wall-clock, tok/s, pooled acceptance,
     mean accept length L, analytic speedup L/(1+0.44), measured speedup vs
     the paired vanilla anchor, switch count + total/mean swap overhead;
     plus a per-language breakdown inside the mixed stream.

Everything is reused from exp1's volume (prompts, LoRAs, head/embed dump):
no fetch/generate/capture/train stages -- this is bench-only and cheap.

Run:
    modal run new/exp2-speedup/pipeline.py::smoke      # 2 langs, 6 prompts
    modal run --detach new/exp2-speedup/pipeline.py::launch
    modal run new/exp2-speedup/pipeline.py::results    # -> results/summary.json
"""
import pathlib

import modal

LOCAL = pathlib.Path(__file__).resolve().parent      # new/exp2-speedup/
ROOT = LOCAL.parent.parent                           # repo root
LORA = ROOT / "lib" / "lora.py"
ONLINE = ROOT / "lib" / "online_dflash.py"
SPEC_PATCH = ROOT / "lib" / "spec_patch.py"

DRAFT_MODEL = "z-lab/Qwen3-8B-DFlash-b16"
TARGET_MODEL = "Qwen/Qwen3-8B"
GPU_BENCH = "H200"           # same GPU as exp1 bench -> numbers comparable
MAX_MODEL_LEN = 2048
C_DFLASH_HF = 0.44           # exp-08 fitted per-step overhead (memory: wallclock-analytic-model)
VARIANTS = ["base", "own", "combined"]

app = modal.App("exp2-speedup")

hf_cache = modal.Volume.from_name("dflash-hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("exp1-language-hidden")   # exp1's volume, reused
VOLS = {"/cache": hf_cache, "/data": data_vol}

PROMPTS = "/data/prompts"            # exp1 stage-1 output (read-only here)
MODELS = "/data/models"              # exp1 stage-5 LoRAs (read-only here)
EXP2 = "/data/exp2"                  # everything this experiment writes
TESTSET = f"{EXP2}/mixed_test.jsonl"
RESULTS = f"{EXP2}/results"

cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy")
    .env({"HF_HOME": "/cache"})
)

# bench image: identical pins to exp1's train/bench image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.9.1", "transformers==4.57.3", "accelerate>=1.0.0",
                 "datasets>=3.0.0", "numpy")
    .env({"HF_HOME": "/cache", "HF_HUB_ENABLE_HF_TRANSFER": "0",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_file(str(ONLINE), "/root/online_dflash.py")
    .add_local_file(str(LORA), "/root/lora.py")
    .add_local_file(str(SPEC_PATCH), "/root/spec_patch.py")
)


# --------------------------------------------------------------------------- #
# helpers (mirrors exp1-language/pipeline.py)
# --------------------------------------------------------------------------- #
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


def _swap_lora(draft, named, sd):
    """Copy a preloaded (CPU) LoRA state dict into the injected LoRALinears.
    This IS the hot-swap being measured -- keep it to the minimal tensor
    copies a real adapter-switching server would do."""
    import torch
    with torch.no_grad():
        for name, entry in sd.items():
            m = named[name]
            m.A.copy_(entry["A"], non_blocking=True)
            m.B.copy_(entry["B"], non_blocking=True)
            m.scaling = float(entry["scaling"])
    torch.cuda.synchronize()


# --------------------------------------------------------------------------- #
# 1) BUILD TESTSET -- eligible langs (>= 1000 train prompts + own LoRA),
#                     n_per_lang test prompts each, one seeded shuffle
# --------------------------------------------------------------------------- #
@app.function(image=cpu_image, timeout=1800, volumes=VOLS)
def build_testset(n_per_lang: int = 25, min_train: int = 1000, seed: int = 0,
                  langs: list = None):
    import json
    import os
    import random

    def _count(path):
        with open(path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    candidates = sorted(langs or os.listdir(PROMPTS))
    eligible, skipped = [], {}
    for lang in candidates:
        train_path = f"{PROMPTS}/{lang}/train.jsonl"
        lora_path = f"{MODELS}/{lang}_lora.pt"
        if not os.path.exists(train_path):
            skipped[lang] = "no train split"
            continue
        n_train = _count(train_path)
        if n_train < min_train:
            skipped[lang] = f"train={n_train} < {min_train}"
            continue
        if not os.path.exists(lora_path):
            skipped[lang] = "no own LoRA"
            continue
        eligible.append(lang)
    assert eligible, f"no eligible languages (skipped: {skipped})"

    items = []
    for lang in eligible:
        with open(f"{PROMPTS}/{lang}/test.jsonl", encoding="utf-8") as f:
            prompts = [json.loads(l)["prompt"] for l in f if l.strip()]
        for p in prompts[:n_per_lang]:
            items.append({"lang": lang, "prompt": p})
    random.Random(seed).shuffle(items)
    for i, it in enumerate(items):
        it["idx"] = i
    switches = sum(1 for a, b in zip(items, items[1:]) if a["lang"] != b["lang"])

    os.makedirs(EXP2, exist_ok=True)
    with open(TESTSET, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    manifest = {"eligible_langs": eligible, "skipped": skipped,
                "n_per_lang": n_per_lang, "min_train": min_train, "seed": seed,
                "n_prompts": len(items), "n_switches": switches}
    with open(f"{EXP2}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    data_vol.commit()
    print(f"[testset] {len(eligible)} langs x {n_per_lang} = {len(items)} "
          f"prompts, {switches} language switches; skipped: {skipped}",
          flush=True)
    return manifest


# --------------------------------------------------------------------------- #
# 2) BENCH STREAM -- one container per variant runs the WHOLE mixed stream
# --------------------------------------------------------------------------- #
@app.function(gpu=GPU_BENCH, image=image, timeout=4 * 3600, volumes=VOLS,
              retries=2)
def bench_stream(variant: str, max_new_tokens: int = 256, limit: int = 0,
                 vanilla_every: int = 20, warmup: int = 2, rank: int = 16):
    import json
    import os
    import sys
    import time

    import torch

    sys.path.insert(0, "/root")
    from spec_patch import make_instrumented_spec_generate

    assert variant in VARIANTS
    with open(f"{EXP2}/manifest.json") as f:
        manifest = json.load(f)
    items = [json.loads(l) for l in open(TESTSET, encoding="utf-8") if l.strip()]
    if limit:
        items = items[:limit]

    tok, target = _load_target()
    draft = _load_draft()

    named = None
    adapters = {}          # lang (or "combined") -> CPU state dict
    if variant != "base":
        from lora import LoRALinear, inject_lora
        inject_lora(draft, rank=rank, alpha=2 * rank,
                    target_modules=("q_proj", "k_proj", "v_proj", "o_proj"))
        draft.to("cuda", dtype=torch.bfloat16)
        named = {n: m for n, m in draft.named_modules()
                 if isinstance(m, LoRALinear)}
        # preload every adapter the stream needs into (pinned) CPU RAM --
        # swap cost is then the honest GPU-upload cost, not disk latency
        keys = (["combined"] if variant == "combined"
                else sorted({it["lang"] for it in items}))
        for key in keys:
            sd = torch.load(f"{MODELS}/{key}_lora.pt", map_location="cpu")
            for entry in sd.values():
                entry["A"] = entry["A"].to(torch.bfloat16).pin_memory()
                entry["B"] = entry["B"].to(torch.bfloat16).pin_memory()
            adapters[key] = sd
        print(f"[bench:{variant}] preloaded {len(adapters)} adapters", flush=True)

    draft.spec_generate = make_instrumented_spec_generate(draft)
    block_size = draft.block_size
    stop_ids = [tok.eos_token_id]

    def build_ids(p):
        return tok([_chat_text(tok, p)], return_tensors="pt").input_ids.to("cuda")

    def first_stop(ids, s):
        for i, t in enumerate(ids):
            if t in s:
                return ids[:i + 1]
        return ids

    if variant == "combined":
        _swap_lora(draft, named, adapters["combined"])

    warm = build_ids(items[0]["prompt"])
    for _ in range(warmup):
        with torch.inference_mode():
            draft.spec_generate(target=target, input_ids=warm, max_new_tokens=64,
                                stop_token_ids=stop_ids, temperature=0.0)
    torch.cuda.synchronize()

    os.makedirs(RESULTS, exist_ok=True)
    out_path = f"{RESULTS}/{variant}.jsonl"
    recs = []
    current_lang = None
    stream_t0 = time.perf_counter()
    with open(out_path, "w") as fout:
        for it in items:
            lang = it["lang"]

            # the dynamic-switching cost under test: own swaps on every
            # language change, base/combined never do
            swap_dt = 0.0
            swapped = False
            if variant == "own" and lang != current_lang:
                t0 = time.perf_counter()
                _swap_lora(draft, named, adapters[lang])
                swap_dt = time.perf_counter() - t0
                current_lang = lang
                swapped = True

            input_ids = build_ids(it["prompt"])
            n_in = input_ids.shape[1]
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.inference_mode():
                out_ids, committed = draft.spec_generate(
                    target=target, input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    stop_token_ids=stop_ids, temperature=0.0)
            torch.cuda.synchronize(); spec_dt = time.perf_counter() - t0
            spec_gen = first_stop(out_ids[0, n_in:].tolist(), stop_ids)
            steps = len(committed)
            gen = sum(committed)
            accepted = sum(c - 1 for c in committed)
            proposed = steps * (block_size - 1)

            rec = {"idx": it["idx"], "lang": lang, "variant": variant,
                   "num_input_tokens": n_in, "forward_steps": steps,
                   "num_generated_tokens": gen,
                   "accepted_draft_tokens": accepted,
                   "proposed_draft_tokens": proposed,
                   "acceptance_rate": (accepted / proposed) if proposed else 0.0,
                   "mean_accept_length": (gen / steps) if steps else 0.0,
                   "spec_seconds": spec_dt, "swap_seconds": swap_dt,
                   "swapped": swapped,
                   "spec_tok_s": len(spec_gen) / spec_dt if spec_dt > 0 else 0.0}

            # paired vanilla decode, same container/GPU, spread across the
            # stream (memory: wallclock-analytic-model)
            if variant == "base" and vanilla_every and it["idx"] % vanilla_every == 0:
                torch.cuda.synchronize(); t0 = time.perf_counter()
                with torch.inference_mode():
                    base_out = target.generate(
                        input_ids=input_ids, max_new_tokens=max_new_tokens,
                        do_sample=False, num_beams=1,
                        pad_token_id=tok.pad_token_id, eos_token_id=stop_ids,
                        use_cache=True)
                torch.cuda.synchronize(); base_dt = time.perf_counter() - t0
                base_gen = first_stop(base_out[0, n_in:].tolist(), stop_ids)
                rec["base_seconds"] = base_dt
                rec["base_tok_s"] = len(base_gen) / base_dt if base_dt > 0 else 0.0

            recs.append(rec)
            fout.write(json.dumps(rec) + "\n")
            if (it["idx"] + 1) % 50 == 0:
                print(f"[bench:{variant}] {it['idx']+1}/{len(items)}", flush=True)
    stream_seconds = time.perf_counter() - stream_t0
    data_vol.commit()

    gen_total = sum(r["num_generated_tokens"] for r in recs)
    work_seconds = sum(r["spec_seconds"] + r["swap_seconds"] for r in recs)
    print(f"[bench:{variant}] stream {stream_seconds:.1f}s "
          f"(spec+swap {work_seconds:.1f}s), {gen_total} tokens, "
          f"{sum(r['swap_seconds'] for r in recs):.2f}s in "
          f"{sum(r['swapped'] for r in recs)} swaps", flush=True)
    return {"variant": variant, "n": len(recs),
            "stream_seconds": stream_seconds, "work_seconds": work_seconds}


# --------------------------------------------------------------------------- #
# 3) AGGREGATE -- stream-level wall clock per variant + per-language breakdown
# --------------------------------------------------------------------------- #
@app.function(image=cpu_image, timeout=1800, volumes=VOLS)
def aggregate():
    import json
    import os

    with open(f"{EXP2}/manifest.json") as f:
        manifest = json.load(f)

    summary = {"manifest": manifest, "variants": {}, "per_lang": {}}
    base_tok_s = None
    for variant in VARIANTS:
        path = f"{RESULTS}/{variant}.jsonl"
        if not os.path.exists(path):
            continue
        recs = [json.loads(l) for l in open(path) if l.strip()]
        if not recs:
            continue
        gen = sum(r["num_generated_tokens"] for r in recs)
        steps = sum(r["forward_steps"] for r in recs)
        acc = (sum(r["accepted_draft_tokens"] for r in recs)
               / max(sum(r["proposed_draft_tokens"] for r in recs), 1))
        L = gen / max(steps, 1)
        spec_s = sum(r["spec_seconds"] for r in recs)
        swap_s = sum(r["swap_seconds"] for r in recs)
        n_swaps = sum(1 for r in recs if r.get("swapped"))
        wall_s = spec_s + swap_s
        row = {"n": len(recs), "acceptance_rate": round(acc, 4),
               "mean_accept_length": round(L, 3),
               "generated_tokens": gen,
               "spec_seconds": round(spec_s, 2),
               "swap_seconds": round(swap_s, 3),
               "wall_seconds": round(wall_s, 2),
               "stream_tok_s": round(gen / wall_s, 2) if wall_s else 0.0,
               "n_swaps": n_swaps,
               "mean_swap_ms": round(1000 * swap_s / n_swaps, 2) if n_swaps else 0.0,
               "swap_overhead_frac": round(swap_s / wall_s, 5) if wall_s else 0.0,
               "speedup_analytic": round(L / (1 + C_DFLASH_HF), 3)}
        if variant == "base":
            v = [r["base_tok_s"] for r in recs if "base_tok_s" in r]
            base_tok_s = (sum(v) / len(v)) if v else None
            if base_tok_s:
                row["base_vanilla_tok_s"] = round(base_tok_s, 2)
        summary["variants"][variant] = row

        by_lang = {}
        for r in recs:
            by_lang.setdefault(r["lang"], []).append(r)
        for lang, rs in sorted(by_lang.items()):
            lacc = (sum(r["accepted_draft_tokens"] for r in rs)
                    / max(sum(r["proposed_draft_tokens"] for r in rs), 1))
            lL = (sum(r["num_generated_tokens"] for r in rs)
                  / max(sum(r["forward_steps"] for r in rs), 1))
            summary["per_lang"].setdefault(lang, {})[variant] = {
                "n": len(rs), "acceptance_rate": round(lacc, 4),
                "mean_accept_length": round(lL, 3),
                "speedup_analytic": round(lL / (1 + C_DFLASH_HF), 3)}

    if base_tok_s:
        for variant, row in summary["variants"].items():
            row["speedup_measured"] = round(row["stream_tok_s"] / base_tok_s, 3)
    with open(f"{RESULTS}/summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    data_vol.commit()
    return summary


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
@app.function(image=cpu_image, timeout=12 * 3600, volumes=VOLS)
def full(n_per_lang: int = 25, min_train: int = 1000, seed: int = 0,
         max_new_tokens: int = 256, limit: int = 0, vanilla_every: int = 20,
         skip_testset: bool = False):
    out = {}
    if not skip_testset:
        out["testset"] = build_testset.remote(n_per_lang=n_per_lang,
                                              min_train=min_train, seed=seed)
        print("[full] testset built", flush=True)
    handles = [bench_stream.spawn(v, max_new_tokens=max_new_tokens,
                                  limit=limit, vanilla_every=vanilla_every)
               for v in VARIANTS]
    out["bench"] = [h.get() for h in handles]
    print("[full] bench done", flush=True)
    out["aggregate"] = aggregate.remote()
    return out


@app.local_entrypoint()
def launch(n_per_lang: int = 25, min_train: int = 1000, seed: int = 0,
           limit: int = 0, skip_testset: bool = False):
    import json
    print(json.dumps(full.remote(
        n_per_lang=n_per_lang, min_train=min_train, seed=seed, limit=limit,
        skip_testset=skip_testset), indent=2, default=str))


@app.local_entrypoint()
def results():
    import json
    summary = aggregate.remote()
    out = LOCAL / "results"
    out.mkdir(exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary["variants"], indent=1))
    print(f"saved -> {out / 'summary.json'}")


@app.local_entrypoint()
def smoke():
    """Tiny end-to-end: 2 languages, 3 prompts each, 64 new tokens (~10 min)."""
    import json
    out = {"testset": build_testset.remote(n_per_lang=3, langs=["French", "Italian"])}
    out["bench"] = [bench_stream.remote(v, max_new_tokens=64, vanilla_every=3,
                                        warmup=1) for v in VARIANTS]
    out["aggregate"] = aggregate.remote()
    print(json.dumps(out, indent=2, default=str))
