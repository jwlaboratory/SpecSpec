#!/usr/bin/env python3
"""Net wall-clock speedup — vanilla target-only decoding baselines.

The spec-decode benches across the repo record `spec_tok_s` per prompt, but only
exp 01/02/06 ever measured the *target-only* baseline, so "does the acceptance
gain survive as wall-clock gain?" is unanswered for the EAGLE (vLLM) and
weird-domains DFlash (HF) matrices. This pipeline fills exactly that hole:

  * vanilla_vllm — one vLLM engine (same build/settings as the EAGLE benches:
    bf16, max_model_len 2048, gpu_mem 0.85, batch-1 generate loop), NO
    speculative_config, over the 8 EAGLE-benched domains (5 languages + 3 weird).
  * vanilla_hf   — HF bf16/sdpa greedy `generate` (same target config as the
    DFlash benches), over the 3 weird domains.

Baselines must match the framework of the spec numbers they divide into: vLLM
and HF have very different batch-1 decode speeds, so a cross-framework ratio
would be meaningless. Greedy vanilla decoding emits the same text as greedy
spec decode (lossless), so token counts line up by construction.

Speedups are then collated OFFLINE from the repo's committed jsonls:
    python experiments/08-wallclock/aggregate.py   # writes results/ + charts

Run:
    modal run --detach experiments/08-wallclock/pipeline.py::launch
    modal volume get --force code-sql-pipeline results/wallclock/ experiments/08-wallclock/results/vanilla/
"""
import pathlib

import modal

LOCAL = pathlib.Path(__file__).resolve().parent            # experiments/08-wallclock/
ROOT = LOCAL.parent.parent                                 # repo root
LANG_DATA = ROOT / "data" / "synthetic"
WEIRD_DATA = ROOT / "experiments" / "03-weird-domains" / "data"

TARGET_MODEL = "Qwen/Qwen3-8B"
GPU = "H200"

LANGS = ["polish", "korean", "italian", "japanese", "german"]
WEIRD = ["translation", "roleplay", "poetry"]

app = modal.App("wallclock-vanilla")

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
    .add_local_dir(str(LANG_DATA), "/root/data/langs")
    .add_local_dir(str(WEIRD_DATA), "/root/data/weird")
)

hf_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.9.1", "transformers==4.57.3", "accelerate>=1.0.0", "numpy")
    .env({"HF_HOME": "/cache", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_dir(str(WEIRD_DATA), "/root/data/weird")
)

hf_cache = modal.Volume.from_name("dflash-hf-cache", create_if_missing=True)
work = modal.Volume.from_name("code-sql-pipeline", create_if_missing=True)
VOLS = {"/cache": hf_cache, "/work": work}

RESULTS = "/work/results/wallclock"


def _read_prompts(path):
    import json
    out = []
    with open(path, encoding="utf-8") as f:
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


def _domain_prompts(key):
    if key.startswith("lang_"):
        return _read_prompts(f"/root/data/langs/{key}/test.jsonl")
    return _read_prompts(f"/root/data/weird/{key}/test.jsonl")


@app.function(gpu=GPU, image=vllm_image, timeout=3 * 3600, volumes=VOLS)
def vanilla_vllm(max_new_tokens: int = 256, limit: int = 100, warmup: int = 2):
    """Target-only vLLM decode over all 8 EAGLE-benched domains, one engine."""
    import json
    import os
    import time

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    llm = LLM(model=TARGET_MODEL, dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=0.85)
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)

    os.makedirs(RESULTS, exist_ok=True)
    out = {}
    for key in [f"lang_{l}" for l in LANGS] + WEIRD:
        texts = [_chat_text(tok, p) for p in _domain_prompts(key)[:limit]]
        for t in texts[:warmup]:
            llm.generate([t], sp)
        recs = []
        t_start = time.time()
        with open(f"{RESULTS}/vllm_{key}.jsonl", "w") as fout:
            for idx, text in enumerate(texts):
                t0 = time.perf_counter()
                outs = llm.generate([text], sp)
                dt = time.perf_counter() - t0
                gen = len(outs[0].outputs[0].token_ids)
                rec = {"domain": key, "prompt_idx": idx, "framework": "vllm",
                       "num_generated_tokens": gen, "seconds": dt,
                       "tok_s": gen / dt if dt > 0 else 0.0}
                fout.write(json.dumps(rec) + "\n")
                recs.append(rec)
        work.commit()
        tot_tok = sum(r["num_generated_tokens"] for r in recs)
        tot_s = sum(r["seconds"] for r in recs)
        out[key] = {"n": len(recs), "pooled_tok_s": tot_tok / tot_s if tot_s else 0.0}
        print(f"[vanilla_vllm:{key}] {out[key]} ({time.time()-t_start:.0f}s)", flush=True)
    return out


@app.function(gpu=GPU, image=hf_image, timeout=3 * 3600, volumes=VOLS)
def vanilla_hf(max_new_tokens: int = 256, limit: int = 100, warmup: int = 2):
    """Target-only HF greedy decode (bf16/sdpa — the DFlash benches' target
    config) over the 3 weird domains."""
    import json
    import os
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda").eval()

    os.makedirs(RESULTS, exist_ok=True)
    out = {}
    for key in WEIRD:
        texts = [_chat_text(tok, p) for p in _domain_prompts(key)[:limit]]
        recs = []
        t_start = time.time()
        with torch.no_grad():
            for t in texts[:warmup]:
                ids = tok(t, return_tensors="pt").input_ids.to("cuda")
                model.generate(ids, max_new_tokens=32, do_sample=False,
                               pad_token_id=tok.eos_token_id)
            with open(f"{RESULTS}/hf_{key}.jsonl", "w") as fout:
                for idx, text in enumerate(texts):
                    ids = tok(text, return_tensors="pt").input_ids.to("cuda")
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    gen_ids = model.generate(ids, max_new_tokens=max_new_tokens,
                                             do_sample=False,
                                             pad_token_id=tok.eos_token_id)
                    torch.cuda.synchronize()
                    dt = time.perf_counter() - t0
                    gen = gen_ids.shape[1] - ids.shape[1]
                    rec = {"domain": key, "prompt_idx": idx, "framework": "hf",
                           "num_generated_tokens": int(gen), "seconds": dt,
                           "tok_s": gen / dt if dt > 0 else 0.0}
                    fout.write(json.dumps(rec) + "\n")
                    recs.append(rec)
                    if idx % 20 == 0:
                        print(f"[vanilla_hf:{key}] {idx+1}/{len(texts)} "
                              f"({time.time()-t_start:.0f}s)", flush=True)
        work.commit()
        tot_tok = sum(r["num_generated_tokens"] for r in recs)
        tot_s = sum(r["seconds"] for r in recs)
        out[key] = {"n": len(recs), "pooled_tok_s": tot_tok / tot_s if tot_s else 0.0}
        print(f"[vanilla_hf:{key}] {out[key]}", flush=True)
    return out


@app.function(image=hf_image, timeout=4 * 3600, volumes=VOLS)
def orchestrate():
    a = vanilla_vllm.spawn()
    b = vanilla_hf.spawn()
    out = {"vllm": a.get(), "hf": b.get()}
    print(f"[orchestrate] {out}", flush=True)
    return out


@app.local_entrypoint()
def launch():
    call = orchestrate.spawn()
    print(f"LAUNCHED orchestrate: {call.object_id}")
