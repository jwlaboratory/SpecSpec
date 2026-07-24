#!/usr/bin/env python3
"""Net spec-decode speedup vs BATCH SIZE, under vLLM continuous batching.

The blog's wall-clock numbers are all batch-1 (single-stream HF `spec_generate`).
But spec-decode speedup is memory-bound-vs-compute-bound sensitive: at batch 1 the
target forward is memory-bound so drafting is nearly free; as concurrency grows the
target saturates compute, the drafter's extra work stops amortizing, and net speedup
collapses toward (or below) 1x. This experiment measures that curve.

Approach (reuses validated in-repo code paths):
  * vLLM native DFlash spec-decode  (exp00 `benchmark_vllm.py`: speculative_config
    method="dflash") — the ONLY path that runs the block-diffusion drafter under
    continuous batching. HF `spec_generate` is hard batch-1.
  * batch-size sweep + tok/s timing  (exp09 `batched-lora-serving`).

Modes per batch size B in {1,2,4,8,16,32,64}:
  * target_only     — Qwen3-8B, no speculation (the baseline throughput at B)
  * dflash          — base DFlash drafter, 15 speculative tokens
  * merged_combined — DFlash with the combined LoRA MERGED into drafter weights
                      (Phase 1b; merged is the only viable serving path under
                      batched spec-decode — unmerged/hot-swap on the drafter is
                      not supported by vLLM's spec path)

speedup(B) = tok_s(mode, B) / tok_s(target_only, B).

Run:
    modal run pipeline.py::sweep                      # full sweep
    modal run pipeline.py::sweep --batch-sizes 1,8,64 # subset
    modal run pipeline.py::smoke                      # tiny 1-container check
"""
from __future__ import annotations

import pathlib

import modal

LOCAL = pathlib.Path(__file__).resolve().parent
ROOT = LOCAL.parent.parent

TARGET_MODEL = "Qwen/Qwen3-8B"
DRAFT_MODEL = "z-lab/Qwen3-8B-DFlash-b16"
NUM_SPEC_TOKENS = 15
GPU = "H200"                       # matches exp-08/11 wall-clock constants
MAX_MODEL_LEN = 2048
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
MODES = ["target_only", "dflash"]  # merged_combined added in Phase 1b

app = modal.App("batchsize-speedup")

hf_cache = modal.Volume.from_name("dflash-hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("exp1-language-hidden")
VOLS = {"/cache": hf_cache, "/data": data_vol}
PROMPTS = "/data/prompts"
MODELS = "/data/models"                 # exp1 LoRA adapters: {name}_lora.pt
MERGED_DIR = "/data/merged_drafters"    # merged DFlash checkpoints for vLLM

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .run_commands(
        "pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly",
        "pip install -U transformers hf_transfer safetensors datasets",
    )
    .env({
        "HF_HOME": "/cache", "CUDA_HOME": "/usr/local/cuda",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN", "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
)

_SPEC_NAMES = (
    "vllm:spec_decode_num_drafts",
    "vllm:spec_decode_num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens",
)
_DIFF_NAMES = (
    "vllm:diffusion_num_committed_tokens",
    "vllm:diffusion_num_canvas_positions",
    "vllm:diffusion_num_denoising_steps",
)


# mode -> which drafter checkpoint to speculate with (None = no speculation).
# "base" uses the stock drafter; merged_* use a checkpoint with the LoRA folded in.
def _drafter_path(mode: str):
    if mode == "target_only":
        return None
    if mode == "base":
        return DRAFT_MODEL
    if mode.startswith("merged_"):
        return f"{MERGED_DIR}/{mode[len('merged_'):]}"   # merged_own_Swedish -> .../own_Swedish
    raise ValueError(f"unknown mode {mode}")


def _build_llm(mode: str, max_num_seqs: int):
    from vllm import LLM
    kw = dict(
        model=TARGET_MODEL,
        trust_remote_code=True,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.90,
        max_num_seqs=max_num_seqs,
        disable_log_stats=False,   # get_metrics() asserts log_stats
    )
    drafter = _drafter_path(mode)
    if drafter is not None:
        kw["speculative_config"] = {
            "method": "dflash", "model": drafter,
            "num_speculative_tokens": NUM_SPEC_TOKENS,
        }
    return LLM(**kw)


@app.function(gpu=GPU, image=vllm_image, timeout=1800, volumes=VOLS)
def merge_drafter(adapter: str, force: bool = False) -> str:
    """Fold {adapter}_lora.pt (exp1 LoRA on q/k/v/o) into the DFlash drafter
    weights and save a standalone checkpoint vLLM can load as its speculative
    model. `adapter` e.g. 'combined' or 'own_Swedish' (-> Swedish_lora.pt).
    Returns the saved dir name suffix used by mode 'merged_<adapter>'."""
    import os
    import shutil
    import torch
    from transformers import AutoModel

    dst = f"{MERGED_DIR}/{adapter}"
    if os.path.exists(f"{dst}/config.json") and not force:
        print(f"[merge] {adapter} already present at {dst}", flush=True)
        return adapter
    lora_name = "combined" if adapter == "combined" else adapter.split("_", 1)[1]
    sd = torch.load(f"{MODELS}/{lora_name}_lora.pt", map_location="cpu")

    draft = AutoModel.from_pretrained(DRAFT_MODEL, trust_remote_code=True,
                                      dtype=torch.bfloat16)
    named = dict(draft.named_modules())
    merged = 0
    with torch.no_grad():
        for mod_name, entry in sd.items():
            m = named[mod_name]                       # nn.Linear in the stock model
            A = entry["A"].to(torch.float32)          # [r, in]
            B = entry["B"].to(torch.float32)          # [out, r]
            dW = float(entry["scaling"]) * (B @ A)    # [out, in]
            m.weight.add_(dW.to(m.weight.dtype))
            merged += 1
    assert merged == len(sd), f"merged {merged} != {len(sd)} adapter modules"

    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst, exist_ok=True)
    draft.save_pretrained(dst)                        # weights + config + auto_map code
    # vLLM loads the drafter's tokenizer too; copy from the source repo cache
    from transformers import AutoTokenizer
    try:
        AutoTokenizer.from_pretrained(DRAFT_MODEL, trust_remote_code=True).save_pretrained(dst)
    except Exception as e:
        print(f"[merge] tokenizer copy skipped: {e}", flush=True)
    data_vol.commit()
    print(f"[merge] {adapter}: folded {merged} modules -> {dst}", flush=True)
    return adapter


@app.function(gpu=GPU, image=vllm_image, timeout=1800, volumes=VOLS)
def check_merged(adapter: str, lang: str, n: int = 32) -> dict:
    """Sanity: run base vs merged_<adapter> on `lang` prompts, report acceptance.
    Confirms the folded LoRA actually shifts behaviour inside vLLM's spec path."""
    from vllm import SamplingParams
    prompts = _render_prompts_lang(lang, n)
    sp = SamplingParams(temperature=0.0, max_tokens=128, min_tokens=128, ignore_eos=True)
    out = {}
    for mode in ["base", f"merged_{adapter}"]:
        llm = _build_llm(mode, max_num_seqs=8)
        before = _snapshot(llm)
        llm.generate(prompts, sp, use_tqdm=False)
        d = {k: _snapshot(llm)[k] - before[k] for k in _snapshot(llm)}
        dt = d.get("vllm:spec_decode_num_draft_tokens", 0.0)
        ac = d.get("vllm:spec_decode_num_accepted_tokens", 0.0)
        out[mode] = round(ac / dt, 4) if dt else None
        print(f"[check {lang}] {mode} acceptance={out[mode]}", flush=True)
        import gc
        import torch
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    return out


def _snapshot(llm) -> dict:
    from vllm.v1.metrics.reader import Counter
    wanted = set(_SPEC_NAMES) | set(_DIFF_NAMES)
    totals = {n: 0.0 for n in wanted}
    for m in llm.get_metrics():
        if isinstance(m, Counter) and m.name in wanted:
            totals[m.name] += m.value
    return totals


MIXED_LANGS = ["English", "French", "German", "Spanish", "Russian", "Chinese",
               "Swedish", "Turkish", "Hungarian", "Korean", "Dutch", "Polish",
               "Hebrew", "Japanese", "Arabic", "Indonesian"]


def _render_prompts_lang(lang: str, n: int) -> list:
    """n rendered WildChat test prompts for `lang` (cycled), Qwen3 chat template.
    lang='mixed' round-robins across MIXED_LANGS to simulate a mixed serving
    stream (what a router would face)."""
    import json
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    raw = []
    if lang == "mixed":
        per = {}
        for l in MIXED_LANGS:
            per[l] = []
            with open(f"{PROMPTS}/{l}/test.jsonl", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        per[l].append(json.loads(line)["prompt"])
        i = 0
        while len(raw) < max(n, len(MIXED_LANGS) * 4):
            l = MIXED_LANGS[i % len(MIXED_LANGS)]
            raw.append(per[l][(i // len(MIXED_LANGS)) % len(per[l])])
            i += 1
    else:
        with open(f"{PROMPTS}/{lang}/test.jsonl", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw.append(json.loads(line)["prompt"])
    rendered = [
        tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
        for p in raw
    ]
    return [rendered[i % len(rendered)] for i in range(n)]


def _render_prompts(n: int) -> list:
    return _render_prompts_lang("English", n)


def _measure(llm, prompts, sp, reps):
    """Return (best tok_s over reps, acceptance, mean_accept_len). Best-of-N
    (not mean) rejects transient slowdowns from neighbours/thermal; steady-state
    throughput has a hard ceiling so the max is the cleanest estimator."""
    import time
    import torch
    tok_s_runs, acc, mal = [], None, None
    for _ in range(reps):
        before = _snapshot(llm)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        after = _snapshot(llm)
        gen = sum(len(o.outputs[0].token_ids) for o in outs)
        tok_s_runs.append(gen / dt if dt else 0.0)
        d = {k: after[k] - before[k] for k in after}
        draft_tok = d.get("vllm:spec_decode_num_draft_tokens", 0.0)
        drafts = d.get("vllm:spec_decode_num_drafts", 0.0)
        accepted = d.get("vllm:spec_decode_num_accepted_tokens", 0.0)
        if draft_tok:
            acc = accepted / draft_tok
            mal = (accepted + drafts) / drafts if drafts else None
    return max(tok_s_runs), tok_s_runs, acc, mal


@app.function(gpu=GPU, image=vllm_image, timeout=2 * 3600, volumes=VOLS,
              max_containers=16, retries=1)
def run_batch(batch_size: int, n_prompts: int = 256, max_tokens: int = 128,
              reps: int = 3) -> dict:
    """BOTH modes for one batch size in ONE container/GPU, so the speedup ratio
    is immune to cross-container GPU variance. Each mode timed reps times
    (best-of-N). Concurrency capped at batch_size; every seq emits exactly
    max_tokens (ignore_eos) -> steady-state decode throughput."""
    import gc
    import torch
    from vllm import SamplingParams

    prompts = _render_prompts(n_prompts)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens,
                        min_tokens=max_tokens, ignore_eos=True)
    out = {"batch_size": batch_size, "n_prompts": n_prompts,
           "max_tokens": max_tokens, "reps": reps}
    for mode in MODES:
        llm = _build_llm(mode, max_num_seqs=batch_size)
        llm.generate(prompts[:batch_size], sp, use_tqdm=False)  # warmup / graphs
        tok_s, runs, acc, mal = _measure(llm, prompts, sp, reps)
        out[mode] = {
            "tok_s": round(tok_s, 2),
            "tok_s_runs": [round(x, 1) for x in runs],
            "acceptance_rate": round(acc, 4) if acc is not None else None,
            "mean_accept_length": round(mal, 3) if mal else None,
        }
        print(f"[bs{batch_size}:{mode}] tok/s={out[mode]['tok_s']} "
              f"(runs {out[mode]['tok_s_runs']}) accept={out[mode]['acceptance_rate']}",
              flush=True)
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    t = out["target_only"]["tok_s"]
    for mode in MODES:
        if mode != "target_only":
            out[mode]["speedup_vs_target"] = round(out[mode]["tok_s"] / t, 3) if t else None
    print(f"[bs{batch_size}] speedup={out.get('dflash',{}).get('speedup_vs_target')}",
          flush=True)
    return out


@app.function(gpu=GPU, image=vllm_image, timeout=3 * 3600, volumes=VOLS, retries=1)
def run_all(batch_sizes: list, max_tokens: int = 128, reps: int = 3) -> list:
    """Whole grid on ONE GPU so every batch size is directly comparable (no
    cross-container variance). Builds/tears down both engines per batch size."""
    import gc
    import torch
    from vllm import SamplingParams

    rows = []
    for b in batch_sizes:
        n = min(512, max(64, 8 * b))
        prompts = _render_prompts(n)
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens,
                            min_tokens=max_tokens, ignore_eos=True)
        out = {"batch_size": b, "n_prompts": n, "max_tokens": max_tokens, "reps": reps}
        for mode in MODES:
            llm = _build_llm(mode, max_num_seqs=b)
            llm.generate(prompts[:b], sp, use_tqdm=False)
            tok_s, runs, acc, mal = _measure(llm, prompts, sp, reps)
            out[mode] = {"tok_s": round(tok_s, 2),
                         "tok_s_runs": [round(x, 1) for x in runs],
                         "acceptance_rate": round(acc, 4) if acc is not None else None,
                         "mean_accept_length": round(mal, 3) if mal else None}
            del llm
            gc.collect()
            torch.cuda.empty_cache()
        t = out["target_only"]["tok_s"]
        out["dflash"]["speedup_vs_target"] = round(out["dflash"]["tok_s"] / t, 3) if t else None
        print(f"[bs{b}] target={t} dflash={out['dflash']['tok_s']} "
              f"speedup={out['dflash']['speedup_vs_target']}", flush=True)
        rows.append(out)
    return rows


@app.function(gpu=GPU, image=vllm_image, timeout=4 * 3600, volumes=VOLS, retries=1)
def run_modes_grid(lang: str, batch_sizes: list, modes: list,
                   max_tokens: int = 128, reps: int = 3) -> list:
    """All modes × all batch sizes on ONE GPU (directly comparable). Each mode is
    target_only / base / merged_own_<lang> / merged_combined; speedup is vs
    target_only at the same batch size. Prompts are <lang>'s test set."""
    import gc
    import torch
    from vllm import SamplingParams

    rows = []
    for b in batch_sizes:
        n = min(512, max(64, 8 * b))
        prompts = _render_prompts_lang(lang, n)
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens,
                            min_tokens=max_tokens, ignore_eos=True)
        out = {"batch_size": b, "n_prompts": n, "lang": lang}
        for mode in modes:
            llm = _build_llm(mode, max_num_seqs=b)
            llm.generate(prompts[:b], sp, use_tqdm=False)
            tok_s, runs, acc, mal = _measure(llm, prompts, sp, reps)
            out[mode] = {"tok_s": round(tok_s, 2),
                         "acceptance_rate": round(acc, 4) if acc is not None else None,
                         "mean_accept_length": round(mal, 3) if mal else None}
            del llm
            gc.collect()
            torch.cuda.empty_cache()
        t = out["target_only"]["tok_s"]
        for mode in modes:
            if mode != "target_only":
                out[mode]["speedup_vs_target"] = round(out[mode]["tok_s"] / t, 3) if t else None
        msg = "  ".join(f"{m}={out[m].get('speedup_vs_target')}" for m in modes if m != "target_only")
        print(f"[bs{b}] {msg}", flush=True)
        rows.append(out)
    return rows


POOL_LANGS = ["English", "French", "German", "Spanish", "Russian", "Chinese",
              "Swedish", "Turkish", "Hungarian", "Korean", "Dutch", "Polish",
              "Hebrew", "Hindi", "Japanese", "Arabic"]


@app.local_entrypoint()
def pooled(langs: str = "", batch: int = 1, reps: int = 3):
    """Overall mean speedup we provide: target/base/merged_combined at one batch
    size across many languages, pooled. Headline number for the blog (batch 1)."""
    import json
    import statistics as st
    ls = [x.strip() for x in langs.split(",") if x.strip()] or POOL_LANGS
    merge_drafter.remote("combined")
    modes = ["target_only", "base", "merged_combined"]
    handles = [(l, run_modes_grid.spawn(l, [batch], modes, reps=reps)) for l in ls]
    rows = []
    for l, h in handles:
        try:
            rows.append(h.get()[0])
        except Exception as e:
            print(f"[{l}] FAILED: {e}", flush=True)
    base_sp = [r["base"]["speedup_vs_target"] for r in rows]
    comb_sp = [r["merged_combined"]["speedup_vs_target"] for r in rows]
    pool = {
        "batch": batch, "n_langs": len(rows),
        "mean_base_speedup_vs_target": round(st.mean(base_sp), 3),
        "mean_combined_speedup_vs_target": round(st.mean(comb_sp), 3),
        "mean_combined_vs_base_pct": round(
            (st.mean(comb_sp) / st.mean(base_sp) - 1) * 100, 2),
        "per_lang": {r["lang"]: {
            "base": r["base"]["speedup_vs_target"],
            "combined": r["merged_combined"]["speedup_vs_target"],
            "base_accept": r["base"]["acceptance_rate"],
            "combined_accept": r["merged_combined"]["acceptance_rate"],
        } for r in rows},
    }
    out = {"target_model": TARGET_MODEL, "gpu": GPU, "modes": modes, "pool": pool}
    dest = LOCAL / "results" / f"pooled_batch{batch}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(pool, indent=1))
    print(f"saved -> {dest}")


@app.local_entrypoint()
def mixed_sweep(batch_sizes: str = "1,4,8,16,32,64", reps: int = 3):
    """Mixed-language serving stream (what a router faces): target/base/
    merged_combined across batch. merged_combined needs no router — one adapter
    serves every language — so this is the deployable mixed-traffic path."""
    import json
    bs = [int(x) for x in batch_sizes.split(",") if x.strip()]
    merge_drafter.remote("combined")
    modes = ["target_only", "base", "merged_combined"]
    rows = run_modes_grid.remote("mixed", bs, modes, reps=reps)
    out = {"target_model": TARGET_MODEL, "draft_model": DRAFT_MODEL,
           "num_spec_tokens": NUM_SPEC_TOKENS, "gpu": GPU, "lang": "mixed",
           "modes": modes, "reps": reps, "rows": rows}
    dest = LOCAL / "results" / "mixed_summary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    print(f"saved -> {dest}")


@app.local_entrypoint()
def modes_sweep(lang: str = "Swedish", batch_sizes: str = "1,4,8,16,32,64",
                reps: int = 3):
    """Unified vLLM cost experiment: net speedup of target/base/merged_own/
    merged_combined across batch sizes, all on one engine + one GPU."""
    import json
    bs = [int(x) for x in batch_sizes.split(",") if x.strip()]
    # ensure the merged drafters exist (idempotent)
    merge_drafter.remote(f"own_{lang}")
    merge_drafter.remote("combined")
    modes = ["target_only", "base", f"merged_own_{lang}", "merged_combined"]
    rows = run_modes_grid.remote(lang, bs, modes, reps=reps)
    out = {"target_model": TARGET_MODEL, "draft_model": DRAFT_MODEL,
           "num_spec_tokens": NUM_SPEC_TOKENS, "gpu": GPU, "lang": lang,
           "modes": modes, "reps": reps, "rows": rows}
    dest = LOCAL / "results" / "modes_summary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    print(f"saved -> {dest}")


@app.local_entrypoint()
def sweep_serial(batch_sizes: str = "", max_tokens: int = 128, reps: int = 3):
    """Single-GPU sweep -> smooth, directly-comparable curve for the blog."""
    import json
    bs = [int(x) for x in batch_sizes.split(",") if x.strip()] or BATCH_SIZES
    rows = run_all.remote(bs, max_tokens=max_tokens, reps=reps)
    out = {"target_model": TARGET_MODEL, "draft_model": DRAFT_MODEL,
           "num_spec_tokens": NUM_SPEC_TOKENS, "gpu": GPU, "serial": True,
           "max_tokens": max_tokens, "reps": reps, "rows": rows}
    dest = LOCAL / "results" / "summary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    print(f"saved -> {dest}")


@app.local_entrypoint()
def sweep(batch_sizes: str = "", n_prompts: int = 256, max_tokens: int = 128,
          reps: int = 3):
    import json
    bs = [int(x) for x in batch_sizes.split(",") if x.strip()] or BATCH_SIZES
    print(f"sweep: {len(bs)} batch sizes {bs} (both modes per container, "
          f"{reps} reps)", flush=True)

    # ~8 waves of steady state per batch size (both modes share n -> fair ratio)
    def n_for(b):
        return min(512, max(64, 8 * b))
    handles = [(b, run_batch.spawn(b, n_prompts=n_for(b), max_tokens=max_tokens,
                                   reps=reps)) for b in bs]
    rows = []
    for b, h in handles:
        try:
            rows.append(h.get())
        except Exception as e:
            print(f"[bs{b}] FAILED: {e}", flush=True)
    rows.sort(key=lambda r: r["batch_size"])

    out = {"target_model": TARGET_MODEL, "draft_model": DRAFT_MODEL,
           "num_spec_tokens": NUM_SPEC_TOKENS, "gpu": GPU,
           "n_prompts": n_prompts, "max_tokens": max_tokens, "reps": reps,
           "rows": rows}
    dest = LOCAL / "results" / "summary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    print(f"saved -> {dest}")


@app.local_entrypoint()
def smoke():
    """One container, tiny, to validate the vLLM DFlash path end to end."""
    print(run_batch.remote(4, n_prompts=16, max_tokens=32, reps=2))


@app.local_entrypoint()
def merge_check(lang: str = "Swedish"):
    """Merge own_<lang> + combined into drafters, then confirm the folded LoRA
    raises acceptance inside vLLM (base < merged) on <lang> prompts."""
    merge_drafter.remote(f"own_{lang}")
    merge_drafter.remote("combined")
    own = check_merged.remote(f"own_{lang}", lang)
    comb = check_merged.remote("combined", lang)
    print("own-merge check:", own)
    print("combined-merge check:", comb)
