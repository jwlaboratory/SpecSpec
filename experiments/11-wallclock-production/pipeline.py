#!/usr/bin/env python3
"""Production wall-clock benchmark for merged vs hot-swapped DFlash LoRAs.

Reuses the WildChat language prompts and LoRAs from `new/exp1-language`, but
fixes the ambiguity in `new/exp2-speedup`:

  * combined LoRA is actually merged into the drafter weights;
  * per-language merged LoRAs are evaluated as already-merged specialists;
  * hot-swapped LoRAs keep the old unmerged-wrapper path for comparison;
  * all modes run sequentially in one H200 container to reduce timing variance.
"""
from __future__ import annotations

import pathlib

import modal

LOCAL = pathlib.Path(__file__).resolve().parent
ROOT = LOCAL.parent.parent
LORA = ROOT / "lib" / "lora.py"
SPEC_PATCH = ROOT / "lib" / "spec_patch.py"

DRAFT_MODEL = "z-lab/Qwen3-8B-DFlash-b16"
TARGET_MODEL = "Qwen/Qwen3-8B"
GPU = "H200"

MODES = ["target_only", "base", "merged_combined", "merged_own", "hotswap_own"]

app = modal.App("wallclock-production-lora")

hf_cache = modal.Volume.from_name("dflash-hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("exp1-language-hidden")
VOLS = {"/cache": hf_cache, "/data": data_vol}

PROMPTS = "/data/prompts"
MODELS = "/data/models"
EXP = "/data/exp11"

cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy")
    .env({"HF_HOME": "/cache"})
)

bench_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.9.1", "transformers==4.57.3", "accelerate>=1.0.0",
                 "datasets>=3.0.0", "numpy")
    .env({
        "HF_HOME": "/cache",
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    .add_local_file(str(LORA), "/root/lora.py")
    .add_local_file(str(SPEC_PATCH), "/root/spec_patch.py")
)


def _root(run_name: str) -> str:
    return f"{EXP}/{run_name}"


def _chat_text(tok, prompt: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _first_stop(ids, stop_ids):
    for i, t in enumerate(ids):
        if t in stop_ids:
            return ids[: i + 1]
    return ids


def _get_child(root, dotted: str):
    mod = root
    for part in dotted.split("."):
        if hasattr(mod, "_modules") and part in mod._modules:
            mod = mod._modules[part]
        else:
            mod = getattr(mod, part)
    return mod


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

    draft = AutoModel.from_pretrained(
        DRAFT_MODEL,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    for p in draft.parameters():
        p.requires_grad_(False)
    return draft


def _load_adapter(path: str, *, pin: bool = False):
    import torch

    sd = torch.load(path, map_location="cpu")
    for entry in sd.values():
        entry["A"] = entry["A"].to(torch.bfloat16)
        entry["B"] = entry["B"].to(torch.bfloat16)
        if pin:
            entry["A"] = entry["A"].pin_memory()
            entry["B"] = entry["B"].pin_memory()
        entry["scaling"] = float(entry["scaling"])
    return sd


def _merge_adapter(draft, sd, *, sign: float = 1.0) -> float:
    """Fold or unfold a LoRA adapter into plain nn.Linear weights.

    Returns wall time for the merge/unmerge operation. This is reported as setup
    time, not included in the already-merged serving tok/s.
    """
    import time
    import torch

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for name, entry in sd.items():
            linear = _get_child(draft, name)
            A = entry["A"].to(device=linear.weight.device, dtype=torch.float32, non_blocking=True)
            B = entry["B"].to(device=linear.weight.device, dtype=torch.float32, non_blocking=True)
            delta = (entry["scaling"] * (B @ A)).to(linear.weight.dtype)
            linear.weight.add_(delta, alpha=sign)
            del A, B, delta
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def _inject_and_swap(draft, named, sd) -> float:
    import time
    import torch

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for name, entry in sd.items():
            m = named[name]
            m.A.copy_(entry["A"], non_blocking=True)
            m.B.copy_(entry["B"], non_blocking=True)
            m.scaling = float(entry["scaling"])
    torch.cuda.synchronize()
    return time.perf_counter() - t0


@app.function(image=cpu_image, timeout=1800, volumes=VOLS)
def build_testset(
    run_name: str = "full",
    n_per_lang: int = 25,
    min_train: int = 1000,
    seed: int = 0,
    langs: list[str] | None = None,
):
    import json
    import os
    import random

    def count(path):
        with open(path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    available = {
        name.lower(): name
        for name in os.listdir(PROMPTS)
        if os.path.isdir(f"{PROMPTS}/{name}")
    }
    if langs:
        candidates = sorted(available.get(lang.lower(), lang) for lang in langs)
    else:
        candidates = sorted(available.values())
    eligible, skipped = [], {}
    for lang in candidates:
        train_path = f"{PROMPTS}/{lang}/train.jsonl"
        test_path = f"{PROMPTS}/{lang}/test.jsonl"
        lora_path = f"{MODELS}/{lang}_lora.pt"
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            skipped[lang] = "missing split"
            continue
        n_train = count(train_path)
        if n_train < min_train:
            skipped[lang] = f"train={n_train} < {min_train}"
            continue
        if not os.path.exists(lora_path):
            skipped[lang] = "missing own LoRA"
            continue
        eligible.append(lang)

    assert eligible, f"no eligible languages; skipped={skipped}"
    items = []
    for lang in eligible:
        with open(f"{PROMPTS}/{lang}/test.jsonl", encoding="utf-8") as f:
            prompts = [json.loads(line)["prompt"] for line in f if line.strip()]
        for p in prompts[:n_per_lang]:
            items.append({"lang": lang, "prompt": p})

    random.Random(seed).shuffle(items)
    for idx, it in enumerate(items):
        it["idx"] = idx
    switches = sum(1 for a, b in zip(items, items[1:]) if a["lang"] != b["lang"])

    root = _root(run_name)
    os.makedirs(f"{root}/results", exist_ok=True)
    with open(f"{root}/mixed_test.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    manifest = {
        "run_name": run_name,
        "eligible_langs": eligible,
        "skipped": skipped,
        "n_per_lang": n_per_lang,
        "min_train": min_train,
        "seed": seed,
        "n_prompts": len(items),
        "n_switches": switches,
        "modes": MODES,
    }
    with open(f"{root}/manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    data_vol.commit()
    print(f"[testset:{run_name}] {len(eligible)} langs x {n_per_lang} = "
          f"{len(items)} prompts; switches={switches}", flush=True)
    return manifest


@app.function(gpu=GPU, image=bench_image, timeout=12 * 3600, volumes=VOLS, retries=1)
def bench_all(
    run_name: str = "full",
    modes: list[str] | None = None,
    max_new_tokens: int = 256,
    limit: int = 0,
    warmup: int = 2,
    rank: int = 16,
):
    import gc
    import json
    import os
    import sys
    import time

    import torch

    sys.path.insert(0, "/root")
    from lora import LoRALinear, inject_lora
    from spec_patch import make_instrumented_spec_generate

    modes = modes or MODES
    for mode in modes:
        assert mode in MODES, f"unknown mode {mode}"

    root = _root(run_name)
    with open(f"{root}/mixed_test.jsonl", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    if limit:
        items = items[:limit]

    os.makedirs(f"{root}/results", exist_ok=True)
    tok, target = _load_target()
    stop_ids = [tok.eos_token_id]

    def ids_for(prompt):
        return tok([_chat_text(tok, prompt)], return_tensors="pt").input_ids.to("cuda")

    def target_one(it):
        input_ids = ids_for(it["prompt"])
        n_in = input_ids.shape[1]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = target.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tok.pad_token_id,
                eos_token_id=stop_ids,
                use_cache=True,
            )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        gen = _first_stop(out[0, n_in:].tolist(), stop_ids)
        return {
            "idx": it["idx"],
            "lang": it["lang"],
            "mode": "target_only",
            "num_input_tokens": n_in,
            "num_generated_tokens": len(gen),
            "seconds": dt,
            "tok_s": len(gen) / dt if dt else 0.0,
        }

    def run_target_only():
        path = f"{root}/results/target_only.jsonl"
        warm_ids = ids_for(items[0]["prompt"])
        for _ in range(warmup):
            with torch.inference_mode():
                target.generate(
                    input_ids=warm_ids,
                    max_new_tokens=min(32, max_new_tokens),
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=stop_ids,
                    use_cache=True,
                )
        torch.cuda.synchronize()
        with open(path, "w", encoding="utf-8") as f:
            for i, it in enumerate(items):
                rec = target_one(it)
                f.write(json.dumps(rec) + "\n")
                if (i + 1) % 50 == 0:
                    print(f"[target_only] {i+1}/{len(items)}", flush=True)
        data_vol.commit()

    def run_spec_mode(mode):
        draft = _load_draft()
        draft.spec_generate = make_instrumented_spec_generate(draft)
        block_size = draft.block_size
        merge_seconds = 0.0
        swap_seconds = 0.0
        n_swaps = 0
        current_lang = None

        named = None
        adapters = {}
        grouped = False

        if mode == "merged_combined":
            sd = _load_adapter(f"{MODELS}/combined_lora.pt")
            merge_seconds += _merge_adapter(draft, sd, sign=1.0)
            del sd
        elif mode == "merged_own":
            grouped = True
        elif mode == "hotswap_own":
            inject_lora(draft, rank=rank, alpha=2 * rank,
                        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"))
            draft.to("cuda", dtype=torch.bfloat16)
            named = {n: m for n, m in draft.named_modules() if isinstance(m, LoRALinear)}
            for lang in sorted({it["lang"] for it in items}):
                adapters[lang] = _load_adapter(f"{MODELS}/{lang}_lora.pt", pin=True)

        def spec_one(it):
            nonlocal swap_seconds, n_swaps, current_lang
            if mode == "hotswap_own" and it["lang"] != current_lang:
                swap_seconds += _inject_and_swap(draft, named, adapters[it["lang"]])
                current_lang = it["lang"]
                n_swaps += 1

            input_ids = ids_for(it["prompt"])
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
            spec_gen = _first_stop(out_ids[0, n_in:].tolist(), stop_ids)
            steps = len(committed)
            gen = sum(committed)
            accepted = sum(c - 1 for c in committed)
            proposed = steps * (block_size - 1)
            return {
                "idx": it["idx"],
                "lang": it["lang"],
                "mode": mode,
                "num_input_tokens": n_in,
                "forward_steps": steps,
                "num_generated_tokens": gen,
                "emitted_tokens": len(spec_gen),
                "accepted_draft_tokens": accepted,
                "proposed_draft_tokens": proposed,
                "acceptance_rate": accepted / proposed if proposed else 0.0,
                "mean_accept_length": gen / steps if steps else 0.0,
                "spec_seconds": dt,
                "spec_tok_s": len(spec_gen) / dt if dt else 0.0,
            }

        warm_ids = ids_for(items[0]["prompt"])
        for _ in range(warmup):
            with torch.inference_mode():
                draft.spec_generate(
                    target=target,
                    input_ids=warm_ids,
                    max_new_tokens=min(64, max_new_tokens),
                    stop_token_ids=stop_ids,
                    temperature=0.0,
                )
        torch.cuda.synchronize()

        path = f"{root}/results/{mode}.jsonl"
        n_done = 0
        with open(path, "w", encoding="utf-8") as f:
            if grouped:
                by_lang = {}
                for it in items:
                    by_lang.setdefault(it["lang"], []).append(it)
                for lang, lang_items in sorted(by_lang.items()):
                    sd = _load_adapter(f"{MODELS}/{lang}_lora.pt")
                    merge_seconds += _merge_adapter(draft, sd, sign=1.0)
                    for it in lang_items:
                        rec = spec_one(it)
                        f.write(json.dumps(rec) + "\n")
                        n_done += 1
                    merge_seconds += _merge_adapter(draft, sd, sign=-1.0)
                    del sd
                    print(f"[{mode}] {n_done}/{len(items)} after {lang}", flush=True)
            else:
                for it in items:
                    rec = spec_one(it)
                    f.write(json.dumps(rec) + "\n")
                    n_done += 1
                    if n_done % 50 == 0:
                        print(f"[{mode}] {n_done}/{len(items)}", flush=True)

        meta = {
            "mode": mode,
            "merge_seconds": merge_seconds,
            "swap_seconds": swap_seconds,
            "n_swaps": n_swaps,
            "grouped_by_lang": grouped,
        }
        with open(f"{root}/results/{mode}.meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
        data_vol.commit()
        del draft, adapters
        gc.collect()
        torch.cuda.empty_cache()

    for mode in modes:
        print(f"[bench_all] START {mode}", flush=True)
        if mode == "target_only":
            run_target_only()
        else:
            run_spec_mode(mode)
        print(f"[bench_all] DONE {mode}", flush=True)

    return {"run_name": run_name, "modes": modes, "n": len(items)}


@app.function(image=cpu_image, timeout=1800, volumes=VOLS)
def aggregate(run_name: str = "full"):
    import json
    import os

    root = _root(run_name)
    out = {"run_name": run_name, "modes": {}, "per_lang": {}}
    manifest_path = f"{root}/manifest.json"
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            out["manifest"] = json.load(f)
    expected_n = out.get("manifest", {}).get("n_prompts")
    out["incomplete_modes"] = {}

    target_tok_s = None
    base_tok_s = None
    for mode in MODES:
        path = f"{root}/results/{mode}.jsonl"
        if not os.path.exists(path):
            continue
        recs = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        if not recs:
            continue
        if expected_n and len(recs) != expected_n:
            out["incomplete_modes"][mode] = {
                "n": len(recs),
                "expected_n": expected_n,
                "path": path,
            }
            continue

        if mode == "target_only":
            gen = sum(r["num_generated_tokens"] for r in recs)
            seconds = sum(r["seconds"] for r in recs)
            row = {
                "n": len(recs),
                "generated_tokens": gen,
                "seconds": round(seconds, 2),
                "tok_s": round(gen / seconds, 2) if seconds else 0.0,
            }
            target_tok_s = gen / seconds if seconds else None
        else:
            gen = sum(r["emitted_tokens"] for r in recs)
            committed = sum(r["num_generated_tokens"] for r in recs)
            steps = sum(r["forward_steps"] for r in recs)
            proposed = sum(r["proposed_draft_tokens"] for r in recs)
            accepted = sum(r["accepted_draft_tokens"] for r in recs)
            spec_seconds = sum(r["spec_seconds"] for r in recs)
            meta = {}
            meta_path = f"{root}/results/{mode}.meta.json"
            if os.path.exists(meta_path):
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
            swap_seconds = float(meta.get("swap_seconds", 0.0))
            merge_seconds = float(meta.get("merge_seconds", 0.0))
            serve_seconds = spec_seconds + swap_seconds
            row = {
                "n": len(recs),
                "generated_tokens": gen,
                "committed_tokens": committed,
                "acceptance_rate": round(accepted / proposed, 4) if proposed else 0.0,
                "mean_accept_length": round(committed / steps, 3) if steps else 0.0,
                "spec_seconds": round(spec_seconds, 2),
                "swap_seconds": round(swap_seconds, 4),
                "merge_seconds_setup": round(merge_seconds, 3),
                "serve_seconds": round(serve_seconds, 2),
                "tok_s": round(gen / serve_seconds, 2) if serve_seconds else 0.0,
                "tok_s_including_merge": round(gen / (serve_seconds + merge_seconds), 2)
                    if serve_seconds + merge_seconds else 0.0,
                "n_swaps": int(meta.get("n_swaps", 0)),
                "grouped_by_lang": bool(meta.get("grouped_by_lang", False)),
            }
            if mode == "base":
                base_tok_s = gen / serve_seconds if serve_seconds else None

        out["modes"][mode] = row

        by_lang = {}
        for r in recs:
            by_lang.setdefault(r["lang"], []).append(r)
        for lang, rs in by_lang.items():
            if mode == "target_only":
                gen = sum(r["num_generated_tokens"] for r in rs)
                sec = sum(r["seconds"] for r in rs)
                cell = {"n": len(rs), "tok_s": round(gen / sec, 2) if sec else 0.0}
            else:
                gen = sum(r["emitted_tokens"] for r in rs)
                sec = sum(r["spec_seconds"] for r in rs)
                committed = sum(r["num_generated_tokens"] for r in rs)
                steps = sum(r["forward_steps"] for r in rs)
                proposed = sum(r["proposed_draft_tokens"] for r in rs)
                accepted = sum(r["accepted_draft_tokens"] for r in rs)
                cell = {
                    "n": len(rs),
                    "tok_s": round(gen / sec, 2) if sec else 0.0,
                    "acceptance_rate": round(accepted / proposed, 4) if proposed else 0.0,
                    "mean_accept_length": round(committed / steps, 3) if steps else 0.0,
                }
            out["per_lang"].setdefault(lang, {})[mode] = cell

    for mode, row in out["modes"].items():
        if target_tok_s and mode != "target_only":
            row["speedup_vs_target"] = round(row["tok_s"] / target_tok_s, 3)
            row["speedup_vs_target_including_merge"] = round(
                row["tok_s_including_merge"] / target_tok_s, 3)
        if base_tok_s and mode not in ("target_only", "base"):
            row["relative_vs_base_dflash"] = round(row["tok_s"] / base_tok_s, 3)

    with open(f"{root}/results/summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    md = [f"# Production wall-clock LoRA serving — {run_name}\n"]
    if "manifest" in out:
        m = out["manifest"]
        md.append(f"{m['n_prompts']} prompts · {len(m['eligible_langs'])} languages · "
                  f"{m['n_switches']} language switches · H200 · greedy decode\n")
    md += [
        "| mode | tok/s | speedup vs target | relative vs base DFlash | accept | L | swaps | setup merge s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        row = out["modes"].get(mode)
        if not row:
            continue
        md.append(
            f"| {mode} | {row.get('tok_s', 0):.2f} | "
            f"{row.get('speedup_vs_target', 1.0):.3f} | "
            f"{row.get('relative_vs_base_dflash', 1.0):.3f} | "
            f"{row.get('acceptance_rate', 0):.4f} | "
            f"{row.get('mean_accept_length', 0):.3f} | "
            f"{row.get('n_swaps', 0)} | "
            f"{row.get('merge_seconds_setup', 0):.3f} |"
        )
    with open(f"{root}/results/report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    data_vol.commit()
    return out


@app.function(image=cpu_image, timeout=14 * 3600, volumes=VOLS)
def full(
    run_name: str = "full",
    n_per_lang: int = 25,
    min_train: int = 1000,
    seed: int = 0,
    max_new_tokens: int = 256,
    limit: int = 0,
    skip_testset: bool = False,
    modes: list[str] | None = None,
):
    if not skip_testset:
        build_testset.remote(run_name=run_name, n_per_lang=n_per_lang,
                             min_train=min_train, seed=seed)
    bench_all.remote(run_name=run_name, modes=modes,
                     max_new_tokens=max_new_tokens, limit=limit)
    return aggregate.remote(run_name=run_name)


@app.local_entrypoint()
def smoke():
    run_name = "smoke"
    build_testset.remote(run_name=run_name, n_per_lang=3, min_train=1000,
                         seed=0, langs=["polish", "korean"])
    bench_all.remote(run_name=run_name, max_new_tokens=64, limit=6, warmup=1)
    out = aggregate.remote(run_name=run_name)
    print(out)


@app.local_entrypoint()
def launch(
    run_name: str = "full",
    n_per_lang: int = 25,
    min_train: int = 1000,
    seed: int = 0,
    max_new_tokens: int = 256,
    limit: int = 0,
    skip_testset: bool = False,
    modes: str = "",
):
    parsed_modes = [mode.strip() for mode in modes.split(",") if mode.strip()] or None
    call = full.spawn(run_name=run_name, n_per_lang=n_per_lang,
                      min_train=min_train, seed=seed,
                      max_new_tokens=max_new_tokens, limit=limit,
                      skip_testset=skip_testset, modes=parsed_modes)
    print(f"LAUNCHED full: {call.object_id}")


@app.local_entrypoint()
def run_blocking(
    run_name: str = "full",
    n_per_lang: int = 25,
    min_train: int = 1000,
    seed: int = 0,
    max_new_tokens: int = 256,
    limit: int = 0,
    skip_testset: bool = False,
    modes: str = "",
):
    parsed_modes = [mode.strip() for mode in modes.split(",") if mode.strip()] or None
    out = full.remote(run_name=run_name, n_per_lang=n_per_lang,
                      min_train=min_train, seed=seed,
                      max_new_tokens=max_new_tokens, limit=limit,
                      skip_testset=skip_testset, modes=parsed_modes)
    print(out)


@app.local_entrypoint()
def results(run_name: str = "full"):
    out = aggregate.remote(run_name=run_name)
    print(out)
