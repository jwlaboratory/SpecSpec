#!/usr/bin/env python3
"""Batch-size scaling of LoRA serving overhead — the hot-swap cost question.

Five serving modes for the SAME rank-16 q/k/v/o adapter recipe on the repo's
independent drafter (Qwen/Qwen3-0.6B), swept over batch sizes:

  A  base model, no LoRA                     (both tracks' in-container baseline)
  B  single UNMERGED LoRA — naive lib/lora.py wrappers (exp 06's serving mode)
  C  single MERGED LoRA — ΔW added into the weights (must time == A)
  D  punica-batched LoRA, ONE adapter shared by every request     [vLLM]
  E  punica-batched LoRA, 50 DISTINCT adapters round-robin        [vLLM]

Track split (exp 08 lesson: only ratio timings within one framework AND one
container): A/B/C run sequentially in ONE HF container; A/D/E each get a vLLM
engine (A twice — once per track — so each ratio has its paired base).

Timing purity: adapters are ZERO-DELTA (lora_B = 0) — punica/SGMV and the
wrapper matmuls do identical work regardless of values, and generation output
is exactly the base model's, so every cell decodes the same tokens. ignore_eos
/ min_new_tokens force exactly MAX_TOKENS per sequence. Prefix caching off.

Run:
    modal run --detach experiments/09-batched-lora-serving/pipeline.py::launch
    modal volume get --force code-sql-pipeline results/batched_lora/ \
        experiments/09-batched-lora-serving/results/raw/
    python experiments/09-batched-lora-serving/make_charts.py
"""
import pathlib

import modal

LOCAL = pathlib.Path(__file__).resolve().parent
ROOT = LOCAL.parent.parent
LORA = ROOT / "lib" / "lora.py"
DATA = ROOT / "data" / "synthetic" / "code_sql"

MODEL = "Qwen/Qwen3-0.6B"
GPU = "H200"
RANK, ALPHA = 16, 32
N_ADAPTERS = 50
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
MAX_TOKENS = 128
N_BATCHES = 3          # timed batches per size (after 1 warmup batch)

app = modal.App("batched-lora-serving")

hf_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.9.1", "transformers==4.57.3", "accelerate>=1.0.0", "numpy")
    .env({"HF_HOME": "/cache", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_file(str(LORA), "/root/lora.py")
    .add_local_dir(str(DATA), "/root/data/code_sql")
)

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .run_commands(
        "pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly",
        "pip install -U transformers hf_transfer safetensors",
    )
    .env({
        "HF_HOME": "/cache", "CUDA_HOME": "/usr/local/cuda",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN", "VLLM_USE_FLASHINFER_SAMPLER": "0",
    })
    .add_local_dir(str(DATA), "/root/data/code_sql")
)

hf_cache = modal.Volume.from_name("dflash-hf-cache", create_if_missing=True)
work = modal.Volume.from_name("code-sql-pipeline", create_if_missing=True)
VOLS = {"/cache": hf_cache, "/work": work}
RESULTS = "/work/results/batched_lora"


def _prompts(tok, n):
    """n chat-formatted prompts, cycling the code_sql test set (100 unique)."""
    import json
    raw = []
    with open("/root/data/code_sql/test.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw.append(json.loads(line)["prompt"])
    out = []
    for i in range(n):
        p = raw[i % len(raw)]
        out.append(tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False))
    return out


def _write(rows_, name):
    import json
    import os
    os.makedirs(RESULTS, exist_ok=True)
    with open(f"{RESULTS}/{name}.jsonl", "w") as f:
        for r in rows_:
            f.write(json.dumps(r) + "\n")
    work.commit()


# --------------------------------------------------------------------------- #
# HF track: A (base) -> B (unmerged wrappers) -> C (merged), one container
# --------------------------------------------------------------------------- #
@app.function(gpu=GPU, image=hf_image, timeout=3 * 3600, volumes=VOLS)
def hf_track():
    import sys
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, "/root")
    from lora import LoRALinear, inject_lora

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda").eval()

    def bench(variant):
        rows_ = []
        with torch.no_grad():
            for bs in BATCH_SIZES:
                batches = [_prompts(tok, bs) for _ in range(N_BATCHES + 1)]
                for bi, batch in enumerate(batches):
                    enc = tok(batch, return_tensors="pt", padding=True).to("cuda")
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    model.generate(**enc, max_new_tokens=MAX_TOKENS,
                                   min_new_tokens=MAX_TOKENS, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
                    torch.cuda.synchronize()
                    dt = time.perf_counter() - t0
                    if bi == 0:
                        continue  # warmup
                    rows_.append({"track": "hf", "variant": variant, "batch_size": bs,
                                  "batch_idx": bi, "gen_tokens": bs * MAX_TOKENS,
                                  "seconds": dt,
                                  "tok_s": bs * MAX_TOKENS / dt})
                print(f"[hf:{variant}] bs={bs} "
                      f"{sum(r['tok_s'] for r in rows_[-N_BATCHES:])/N_BATCHES:.0f} tok/s",
                      flush=True)
        return rows_

    rows_ = bench("A_base")

    # B: naive unmerged wrappers (zero delta: B matrices init to 0 already)
    inject_lora(model, rank=RANK, alpha=ALPHA,
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"))
    model.to("cuda", dtype=torch.bfloat16)
    rows_ += bench("B_unmerged")

    # C: merged — fold ΔW (=0 here; the fold is exact regardless) back into the
    # base linears and REMOVE the wrappers, restoring plain nn.Linear modules.
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, LoRALinear):
                with torch.no_grad():
                    child.base.weight += child.delta_weight().to(child.base.weight.dtype)
                setattr(mod, child_name, child.base)
    rows_ += bench("C_merged")

    _write(rows_, "hf_track")
    return {"rows": len(rows_)}


# --------------------------------------------------------------------------- #
# vLLM track: one engine per variant (A / D / E), paired base in-track
# --------------------------------------------------------------------------- #
def _make_peft_adapters(n, out_root="/root/adapters"):
    """n zero-delta rank-16 q/k/v/o PEFT adapter dirs for vLLM's punica path."""
    import json
    import os

    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(MODEL)
    hidden = cfg.hidden_size
    head_dim = getattr(cfg, "head_dim", hidden // cfg.num_attention_heads)
    q_out = cfg.num_attention_heads * head_dim
    kv_out = cfg.num_key_value_heads * head_dim
    shapes = {"q_proj": (hidden, q_out), "k_proj": (hidden, kv_out),
              "v_proj": (hidden, kv_out), "o_proj": (q_out, hidden)}

    paths = []
    gen = torch.Generator().manual_seed(0)
    for a in range(n):
        d = f"{out_root}/dom{a}"
        os.makedirs(d, exist_ok=True)
        sd = {}
        for layer in range(cfg.num_hidden_layers):
            for proj, (in_f, out_f) in shapes.items():
                pre = f"base_model.model.model.layers.{layer}.self_attn.{proj}"
                sd[f"{pre}.lora_A.weight"] = torch.randn(
                    (RANK, in_f), generator=gen, dtype=torch.float32) * 0.01
                sd[f"{pre}.lora_B.weight"] = torch.zeros((out_f, RANK))
        save_file({k: v.to(torch.bfloat16).contiguous() for k, v in sd.items()},
                  os.path.join(d, "adapter_model.safetensors"))
        with open(os.path.join(d, "adapter_config.json"), "w") as f:
            json.dump({"peft_type": "LORA", "r": RANK, "lora_alpha": ALPHA,
                       "lora_dropout": 0.0, "bias": "none",
                       "base_model_name_or_path": MODEL,
                       "task_type": "CAUSAL_LM", "fan_in_fan_out": False,
                       "modules_to_save": None,
                       "target_modules": list(shapes)}, f)
        paths.append(d)
    return paths


@app.function(gpu=GPU, image=vllm_image, timeout=3 * 3600, volumes=VOLS)
def vllm_track(variant: str):
    import time

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    assert variant in ("A_base", "D_punica_1", "E_punica_50")
    n_adapters = {"A_base": 0, "D_punica_1": 1, "E_punica_50": N_ADAPTERS}[variant]

    kwargs = dict(model=MODEL, dtype="bfloat16", max_model_len=2048,
                  gpu_memory_utilization=0.85, max_num_seqs=max(BATCH_SIZES),
                  enable_prefix_caching=False)
    if n_adapters:
        kwargs.update(enable_lora=True, max_loras=n_adapters, max_lora_rank=RANK,
                      max_cpu_loras=n_adapters)
    llm = LLM(**kwargs)
    tok = AutoTokenizer.from_pretrained(MODEL)
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, ignore_eos=True)

    lora_reqs = None
    if n_adapters:
        from vllm.lora.request import LoRARequest
        paths = _make_peft_adapters(n_adapters)
        lora_reqs = [LoRARequest(f"dom{i}", i + 1, p) for i, p in enumerate(paths)]

    def bench(label, use_lora):
        rows_ = []
        for bs in BATCH_SIZES:
            for bi in range(N_BATCHES + 1):
                batch = _prompts(tok, bs)
                reqs = ([lora_reqs[i % n_adapters] for i in range(bs)]
                        if use_lora else None)
                t0 = time.perf_counter()
                llm.generate(batch, sp, lora_request=reqs)
                dt = time.perf_counter() - t0
                if bi == 0:
                    continue  # warmup (includes first-touch adapter load)
                rows_.append({"track": "vllm", "variant": label, "batch_size": bs,
                              "batch_idx": bi, "gen_tokens": bs * MAX_TOKENS,
                              "seconds": dt, "tok_s": bs * MAX_TOKENS / dt})
            print(f"[vllm:{label}] bs={bs} "
                  f"{sum(r['tok_s'] for r in rows_[-N_BATCHES:])/N_BATCHES:.0f} tok/s",
                  flush=True)
        return rows_

    rows_ = []
    if n_adapters:
        # paired in-container base: same LoRA-enabled engine, adapter idle
        rows_ += bench(f"{variant}_nolora", use_lora=False)
        rows_ += bench(variant, use_lora=True)
    else:
        rows_ += bench(variant, use_lora=False)
    _write(rows_, f"vllm_{variant}")
    return {"rows": len(rows_)}


@app.function(image=hf_image, timeout=4 * 3600, volumes=VOLS)
def orchestrate():
    jobs = {"hf": hf_track.spawn()}
    for v in ("A_base", "D_punica_1", "E_punica_50"):
        jobs[v] = vllm_track.spawn(v)
    out = {k: c.get() for k, c in jobs.items()}
    print(f"[orchestrate] {out}", flush=True)
    return out


@app.local_entrypoint()
def launch():
    call = orchestrate.spawn()
    print(f"LAUNCHED orchestrate: {call.object_id}")
