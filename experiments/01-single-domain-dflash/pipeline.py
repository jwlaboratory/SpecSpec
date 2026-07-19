#!/usr/bin/env python3
"""Per-domain DFlash drafter experiment on Modal: base vs full fine-tune vs LoRA.

Generalized over any domain under `data/downloaded/<domain>/`
(train/val/test.jsonl of prompts). ONE drafter stack, REAL weights:
`z-lab/Qwen3-8B-DFlash-b16` (the pretrained DFlash drafter) is loaded via AutoModel
and used for BOTH training and benchmarking — it is architecturally identical to
SpecForge's `DFlashDraftModel`, so we train it with the vendored SpecForge
`OnlineDFlashModel` (DFlash matching loss) and benchmark it with its own
`spec_generate`. No second model, no weight remapping.

Self-distillation: the frozen Qwen3-8B TARGET generates the answers and the drafter
learns to match the TARGET's own tokens (even where the target is wrong) using the
target's LIVE hidden states at the drafter's target layers. The dataset's own answers
are NOT used — only its prompts.

Stages (A100-80GB):
  1. prep   — target generates a response per prompt; cache token ids.
  2. train  — fine-tune the drafter two ways: `lora` (q/k/v/o, base frozen) and
              `full` (every drafter weight). DFlash exp-weighted block CE vs the
              target's tokens, target hidden states computed live each step.
  3. bench  — base / full / lora via the real spec_generate + a target-only
              baseline over the held-out TEST split; acceptance rate, mean accept
              length, speedup, exact-match. Aggregated into one comparison table.

Run one domain end-to-end:
    modal run "Trained Models/pipeline.py" --domain ood_indian_legal --epochs 3

Outputs land on the `code-sql-pipeline` volume under /models/<domain>_{lora,full}.pt
and /results/<domain>/. The local entrypoint prints the pull commands.
"""
import pathlib

import modal

LOCAL = pathlib.Path(__file__).resolve().parent          # experiments/01-single-domain-dflash/
ROOT = LOCAL.parent.parent                               # repo root
LORA = ROOT / "lib" / "lora.py"
SPEC_PATCH = ROOT / "lib" / "spec_patch.py"
DOWNLOADED = ROOT / "data" / "downloaded"                # data/downloaded/<domain>/

DRAFT_MODEL = "z-lab/Qwen3-8B-DFlash-b16"
TARGET_MODEL = "Qwen/Qwen3-8B"
GPU = "H200"   # 141GB, ~1.5-2x an A100 — faster vLLM prep + room for a big train batch

app = modal.App("dflash-domain-pipeline")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.9.1", "transformers==4.57.3", "accelerate>=1.0.0",
                 "datasets>=3.0.0", "numpy")
    .env({"HF_HOME": "/cache", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_file(str(ROOT / "lib" / "online_dflash.py"), "/root/online_dflash.py")
    .add_local_file(str(LORA), "/root/lora.py")
    .add_local_file(str(SPEC_PATCH), "/root/spec_patch.py")
    .add_local_dir(str(DOWNLOADED), "/root/downloaded")
)

# Separate image JUST for prep: vLLM (continuous batching + paged attention) makes
# target generation ~10-30x faster than HF generate(). Same recipe the repo's vLLM
# benchmark uses (CUDA devel base for nvcc, vLLM nightly, FLASH_ATTN). vLLM brings
# its own torch, so it stays isolated from the train/bench image (torch 2.9.1 +
# transformers 4.57.3, which the z-lab remote code + OnlineDFlashModel need).
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
    .add_local_dir(str(DOWNLOADED), "/root/downloaded")
)

hf_cache = modal.Volume.from_name("dflash-hf-cache", create_if_missing=True)
work = modal.Volume.from_name("code-sql-pipeline", create_if_missing=True)
VOLS = {"/cache": hf_cache, "/work": work}


def _read_prompts(domain, split):
    import json
    out = []
    with open(f"/root/downloaded/{domain}/{split}.jsonl", encoding="utf-8") as f:
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
# 1) PREP
# --------------------------------------------------------------------------- #
@app.function(gpu=GPU, image=vllm_image, timeout=3600, volumes=VOLS)
def prep(domain: str, max_new_tokens: int = 256, limit: int = 0, gen_batch: int = 0):
    """Generate a target answer per prompt with vLLM (continuous batching) and cache
    the token ids. gen_batch is ignored (vLLM schedules everything internally)."""
    import os
    import time

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    llm = LLM(model=TARGET_MODEL, dtype="bfloat16", gpu_memory_utilization=0.90,
              max_model_len=2048, enable_prefix_caching=True)
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    os.makedirs(f"/work/prep/{domain}", exist_ok=True)
    summary = {}
    for split in ("train", "val"):
        prompts = _read_prompts(domain, split)
        if limit:
            prompts = prompts[:limit]
        texts = [_chat_text(tok, p) for p in prompts]
        t0 = time.time()
        outs = llm.generate(texts, sp)   # one call — vLLM batches all of them
        records = []
        for o in outs:
            pids = list(o.prompt_token_ids)
            gids = list(o.outputs[0].token_ids)
            if len(gids) < 2:
                continue
            records.append({"input_ids": torch.tensor(pids + gids, dtype=torch.long),
                            "prompt_len": len(pids)})
        torch.save(records, f"/work/prep/{domain}/{split}.pt")
        resp = [len(r["input_ids"]) - r["prompt_len"] for r in records]
        summary[split] = {
            "n": len(records), "sec": round(time.time() - t0, 1),
            "mean_resp_len": round(sum(resp) / max(len(resp), 1), 1),
            "mean_total_len": round(sum(len(r["input_ids"]) for r in records) / max(len(records), 1), 1)}
        print(f"[prep:{domain}:{split}] {summary[split]}", flush=True)
    work.commit()
    return summary


# --------------------------------------------------------------------------- #
# 2) TRAIN
# --------------------------------------------------------------------------- #
@app.function(gpu=GPU, image=image, timeout=6 * 3600, volumes=VOLS)
def train_variant(domain: str, mode: str, epochs: int = 3, lr: float = 0.0,
                  batch_size: int = 12, num_anchors: int = 48, max_seq_len: int = 512,
                  val_every: int = 200):
    import os
    import sys
    import time

    import torch

    sys.path.insert(0, "/root")
    from online_dflash import OnlineDFlashModel

    assert mode in ("lora", "full")
    if lr == 0.0:
        lr = 1e-3 if mode == "lora" else 1e-5

    tok, target = _load_target()
    draft = _load_draft()
    extract = sys.modules[type(draft).__module__].extract_context_feature
    target_layer_ids = draft.target_layer_ids
    mask_token_id = draft.mask_token_id
    block_size = draft.block_size

    if mode == "lora":
        from lora import inject_lora, lora_trainable_parameters
        replaced = inject_lora(draft, rank=16, alpha=32,
                               target_modules=("q_proj", "k_proj", "v_proj", "o_proj"))
        draft.to("cuda", dtype=torch.bfloat16)
        trainable = list(lora_trainable_parameters(draft))
        print(f"[train:{domain}:lora] {len(replaced)} layers, "
              f"{sum(p.numel() for p in trainable):,} params", flush=True)
    else:
        for p in draft.parameters():
            p.requires_grad_(True)
        trainable = [p for p in draft.parameters() if p.requires_grad]
        print(f"[train:{domain}:full] {sum(p.numel() for p in trainable):,} params", flush=True)

    online = OnlineDFlashModel(
        draft_model=draft, target_lm_head=target.lm_head,
        target_embed_tokens=target.get_input_embeddings(),
        mask_token_id=mask_token_id, block_size=block_size,
        attention_backend="sdpa", num_anchors=num_anchors,
        loss_decay_gamma=7.0, loss_type="dflash",
    )
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)

    train_data = torch.load(f"/work/prep/{domain}/train.pt")
    val_data = torch.load(f"/work/prep/{domain}/val.pt")
    print(f"[train:{domain}:{mode}] {len(train_data)} train / {len(val_data)} val", flush=True)

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
        for i in range(0, min(len(val_data), 60), batch_size):
            r = run_batch(val_data[i : i + batch_size], train=False)
            if r:
                ls.append(r[0]); accs.append(r[1])
        online.train()
        return (sum(ls) / len(ls), sum(accs) / len(accs)) if ls else (None, None)

    log = {"domain": domain, "mode": mode, "lr": lr, "epochs": epochs, "val": []}
    online.train()
    step = 0
    t0 = time.time()
    log["val_initial"] = validate()
    print(f"[train:{domain}:{mode}] initial val {log['val_initial']}", flush=True)
    for ep in range(epochs):
        order = torch.randperm(len(train_data)).tolist()
        for i in range(0, len(train_data), batch_size):
            r = run_batch([train_data[j] for j in order[i : i + batch_size]], train=True)
            if r is None:
                continue
            step += 1
            if step % 50 == 0:
                print(f"[train:{domain}:{mode}] ep{ep} step{step} loss={r[0]:.4f} "
                      f"acc={r[1]:.4f} ({time.time()-t0:.0f}s)", flush=True)
            if step % val_every == 0:
                vl, va = validate()
                log["val"].append({"step": step, "loss": vl, "acc": va})
                print(f"[train:{domain}:{mode}]   val step{step} loss={vl:.4f} acc={va:.4f}", flush=True)
    log["val_final"] = validate()
    print(f"[train:{domain}:{mode}] final val {log['val_final']}", flush=True)

    os.makedirs("/work/models", exist_ok=True)
    if mode == "lora":
        from lora import lora_state_dict
        path = f"/work/models/{domain}_lora.pt"
        torch.save(lora_state_dict(draft), path)
    else:
        path = f"/work/models/{domain}_full.pt"
        torch.save({k: v.to(torch.bfloat16).cpu() for k, v in draft.state_dict().items()}, path)
    work.commit()
    log["saved"] = path
    log["train_seconds"] = round(time.time() - t0, 1)
    print(f"[train:{domain}:{mode}] saved -> {path}", flush=True)
    return log


# --------------------------------------------------------------------------- #
# 3) BENCH
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
def bench_variant(domain: str, variant: str, max_new_tokens: int = 256,
                  limit: int = 100, warmup: int = 2):
    import json
    import os
    import sys
    import time

    import torch

    sys.path.insert(0, "/root")
    from spec_patch import make_instrumented_spec_generate

    assert variant in ("base_dflash", "full_finetune", "lora")
    tok, target = _load_target()
    draft = _load_draft()

    if variant == "full_finetune":
        sd = torch.load(f"/work/models/{domain}_full.pt", map_location="cuda")
        missing, unexpected = draft.load_state_dict(sd, strict=False)
        print(f"[bench:{domain}:{variant}] loaded full (missing={len(missing)} unexpected={len(unexpected)})", flush=True)
    elif variant == "lora":
        from lora import inject_lora
        inject_lora(draft, rank=16, alpha=32,
                    target_modules=("q_proj", "k_proj", "v_proj", "o_proj"))
        draft.to("cuda", dtype=torch.bfloat16)
        _load_lora_into(draft, torch.load(f"/work/models/{domain}_lora.pt", map_location="cuda"))
        print(f"[bench:{domain}:{variant}] injected + loaded LoRA", flush=True)

    draft.spec_generate = make_instrumented_spec_generate(draft)
    block_size = draft.block_size
    stop_ids = [tok.eos_token_id]
    prompts = _read_prompts(domain, "test")[:limit]

    def build_ids(p):
        return tok([_chat_text(tok, p)], return_tensors="pt").input_ids.to("cuda")

    def first_stop(ids, s):
        for i, t in enumerate(ids):
            if t in s:
                return ids[: i + 1]
        return ids

    def summarize(committed):
        steps = len(committed)
        gen = sum(committed)
        accepted = sum(c - 1 for c in committed)
        proposed = steps * (block_size - 1)
        return {"forward_steps": steps, "num_generated_tokens": gen,
                "accepted_draft_tokens": accepted, "proposed_draft_tokens": proposed,
                "acceptance_rate": (accepted / proposed) if proposed else 0.0,
                "mean_accept_length": (gen / steps) if steps else 0.0}

    warm = build_ids("Explain a legal concept in one sentence.")
    for _ in range(warmup):
        with torch.inference_mode():
            draft.spec_generate(target=target, input_ids=warm, max_new_tokens=64,
                                stop_token_ids=stop_ids, temperature=0.0)
    torch.cuda.synchronize()

    os.makedirs(f"/work/results/{domain}", exist_ok=True)
    out_path = f"/work/results/{domain}/{variant}.jsonl"
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
            stats = summarize(committed)

            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.inference_mode():
                base_out = target.generate(
                    input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                    num_beams=1, pad_token_id=tok.pad_token_id, eos_token_id=stop_ids, use_cache=True)
            torch.cuda.synchronize(); base_dt = time.perf_counter() - t0
            base_gen = first_stop(base_out[0, n_in:].tolist(), stop_ids)
            base_tok_s = len(base_gen) / base_dt if base_dt > 0 else 0.0
            m = min(len(spec_gen), len(base_gen))
            divergence = next((i for i in range(m) if spec_gen[i] != base_gen[i]), m)

            rec = {"category": variant, "prompt_idx": idx, "num_input_tokens": n_in,
                   "spec_seconds": spec_dt, "spec_tok_s": spec_tok_s, **stats,
                   "committed_per_step": committed, "baseline_seconds": base_dt,
                   "baseline_tok_s": base_tok_s, "baseline_gen_tokens": len(base_gen),
                   "speedup": (spec_tok_s / base_tok_s) if base_tok_s > 0 else 0.0,
                   "exact_match": bool(spec_gen == base_gen), "first_divergence": divergence,
                   "agreement_frac": (divergence / m) if m else 1.0}
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            recs.append(rec)
            if idx % 10 == 0:
                print(f"[bench:{domain}:{variant}] {idx+1}/{len(prompts)} "
                      f"accept={stats['acceptance_rate']*100:.0f}% "
                      f"mean_len={stats['mean_accept_length']:.2f} "
                      f"speedup={rec['speedup']:.2f}x ({time.time()-t_start:.0f}s)", flush=True)
    work.commit()

    def mean(k):
        return sum(r[k] for r in recs) / len(recs)
    tot_acc = sum(r["accepted_draft_tokens"] for r in recs)
    tot_prop = sum(r["proposed_draft_tokens"] for r in recs)
    summ = {"domain": domain, "variant": variant, "n": len(recs),
            "acceptance_rate_pooled": (tot_acc / tot_prop) if tot_prop else 0.0,
            "mean_accept_length": mean("mean_accept_length"), "spec_tok_s": mean("spec_tok_s"),
            "baseline_tok_s": mean("baseline_tok_s"), "speedup": mean("speedup"),
            "exact_match_rate": sum(bool(r["exact_match"]) for r in recs) / len(recs)}
    print(f"[bench:{domain}:{variant}] DONE {summ}", flush=True)
    return summ


# --------------------------------------------------------------------------- #
# aggregate
# --------------------------------------------------------------------------- #
@app.function(image=image, timeout=1200, volumes=VOLS)
def aggregate(domain: str):
    import json

    work.reload()
    variants = ["base_dflash", "full_finetune", "lora"]
    labels = {"base_dflash": "base (pretrained)", "full_finetune": "full fine-tune", "lora": "LoRA"}
    rdir = f"/work/results/{domain}"
    rows = {}
    merged = f"{rdir}/{domain}.jsonl"
    with open(merged, "w") as fout:
        for v in variants:
            try:
                recs = []
                with open(f"{rdir}/{v}.jsonl") as f:
                    for line in f:
                        if line.strip():
                            recs.append(json.loads(line)); fout.write(line)
            except FileNotFoundError:
                continue
            if not recs:
                continue
            n = len(recs)
            tp = sum(r["proposed_draft_tokens"] for r in recs)
            rows[v] = {"variant": v, "n": n,
                       "acceptance_rate_pooled": (sum(r["accepted_draft_tokens"] for r in recs) / tp) if tp else 0.0,
                       "mean_accept_length": sum(r["mean_accept_length"] for r in recs) / n,
                       "spec_tok_s": sum(r["spec_tok_s"] for r in recs) / n,
                       "baseline_tok_s": sum(r["baseline_tok_s"] for r in recs) / n,
                       "speedup": sum(r["speedup"] for r in recs) / n,
                       "exact_match_rate": sum(bool(r["exact_match"]) for r in recs) / n,
                       "mean_forward_steps": sum(r["forward_steps"] for r in recs) / n}

    cols = ["variant", "n", "acceptance_rate_pooled", "mean_accept_length",
            "mean_forward_steps", "spec_tok_s", "baseline_tok_s", "speedup", "exact_match_rate"]
    csv_lines = [",".join(cols)]
    for v in variants:
        if v in rows:
            csv_lines.append(",".join(
                f"{rows[v][c]:.4f}" if isinstance(rows[v][c], float) else str(rows[v][c]) for c in cols))
    csv_text = "\n".join(csv_lines) + "\n"

    base = rows.get("base_dflash")

    def signed(diff, unit="", nd=2):
        return " (" + ("+" if diff >= 0 else "") + format(diff, "." + str(nd) + "f") + unit + ")"

    md = [f"# {domain} — base vs full fine-tune vs LoRA (DFlash drafter)\n",
          f"Domain: **{domain}** · target **Qwen/Qwen3-8B** · drafter "
          "**z-lab/Qwen3-8B-DFlash-b16** · temperature 0 (lossless) · held-out test split.\n",
          "Metric that matters: **mean accept length** (tokens committed per target "
          "pass) and **acceptance rate** — higher = the drafter tracks the target "
          "better on this domain.\n",
          "| variant | n | accept rate | mean accept len | fwd steps | spec tok/s | speedup | exact-match |",
          "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for v in variants:
        if v not in rows:
            continue
        d = rows[v]
        is_base = (v == "base_dflash") or (base is None)
        acc = "{:.1f}%".format(d["acceptance_rate_pooled"] * 100)
        mal = "{:.2f}".format(d["mean_accept_length"])
        spd = "{:.2f}x".format(d["speedup"])
        if not is_base:
            acc += signed((d["acceptance_rate_pooled"] - base["acceptance_rate_pooled"]) * 100, "pp", 1)
            mal += signed(d["mean_accept_length"] - base["mean_accept_length"])
            spd += signed(d["speedup"] - base["speedup"])
        md.append("| {label} | {n} | {acc} | {mal} | {fwd:.1f} | {spec:.1f} | {spd} | {ex:.0f}% |".format(
            label=labels[v], n=d["n"], acc=acc, mal=mal, fwd=d["mean_forward_steps"],
            spec=d["spec_tok_s"], spd=spd, ex=d["exact_match_rate"] * 100))
    md.append("\n_exact-match = spec output equals the target's own greedy output "
              "(temperature 0 ⇒ DFlash is lossless; the adapter changes speed, not "
              "correctness). Deltas in parentheses are vs. the base drafter._\n")
    md_text = "\n".join(md)

    with open(f"{rdir}/{domain}_comparison.csv", "w") as f:
        f.write(csv_text)
    with open(f"{rdir}/{domain}_report.md", "w") as f:
        f.write(md_text)
    work.commit()
    print("\n" + md_text)
    return {"_report.md": md_text, "_comparison.csv": csv_text, "rows": rows}


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
@app.local_entrypoint()
def main(domain: str = "ood_indian_legal", epochs: int = 3, prep_gen_batch: int = 32,
         prep_max_new_tokens: int = 256, bench_max_new_tokens: int = 256,
         bench_limit: int = 150, skip_prep: bool = False):
    import json

    if not skip_prep:
        print(f"=== STAGE 1: prep [{domain}] (target generates responses) ===")
        print(json.dumps(prep.remote(domain, max_new_tokens=prep_max_new_tokens,
                                     gen_batch=prep_gen_batch), indent=2))

    print(f"\n=== STAGE 2: train lora + full [{domain}] (parallel) ===")
    lc = train_variant.spawn(domain, "lora", epochs=epochs)
    fc = train_variant.spawn(domain, "full", epochs=epochs)
    ll, fl = lc.get(), fc.get()
    print("[lora]", ll.get("val_final"), "->", ll.get("saved"))
    print("[full]", fl.get("val_final"), "->", fl.get("saved"))

    print(f"\n=== STAGE 3: bench base / full / lora [{domain}] (parallel) ===")
    calls = {v: bench_variant.spawn(domain, v, max_new_tokens=bench_max_new_tokens, limit=bench_limit)
             for v in ("base_dflash", "full_finetune", "lora")}
    for v, c in calls.items():
        print(v, json.dumps(c.get(), indent=2))

    print(f"\n=== aggregate [{domain}] ===")
    print(aggregate.remote(domain)["_report.md"])


@app.function(image=image, timeout=10 * 3600, volumes=VOLS)
def orchestrate(domain: str, epochs: int = 3, prep_gen_batch: int = 32,
                prep_max_new_tokens: int = 256, bench_max_new_tokens: int = 256,
                bench_limit: int = 150, skip_prep: bool = False):
    """SERVER-SIDE orchestration of the whole pipeline. Runs inside a (cheap, CPU)
    Modal container and drives the GPU stages, so a LOCAL network drop can't kill
    the run. Launch detached:
        modal run --detach experiments/01-single-domain-dflash/pipeline.py::run --domain <domain>
    """
    out = {"domain": domain}
    if not skip_prep:
        out["prep"] = prep.remote(domain, max_new_tokens=prep_max_new_tokens, gen_batch=prep_gen_batch)
        print(f"[orchestrate:{domain}] prep done: {out['prep']}", flush=True)
    lc = train_variant.spawn(domain, "lora", epochs=epochs)
    fc = train_variant.spawn(domain, "full", epochs=epochs)
    ll, fl = lc.get(), fc.get()
    out["train"] = {"lora": ll.get("val_final"), "full": fl.get("val_final"),
                    "lora_saved": ll.get("saved"), "full_saved": fl.get("saved")}
    print(f"[orchestrate:{domain}] train done: {out['train']}", flush=True)
    calls = {v: bench_variant.spawn(domain, v, max_new_tokens=bench_max_new_tokens, limit=bench_limit)
             for v in ("base_dflash", "full_finetune", "lora")}
    out["bench"] = {v: c.get() for v, c in calls.items()}
    print(f"[orchestrate:{domain}] bench done", flush=True)
    agg = aggregate.remote(domain)
    out["report"] = agg["_report.md"]
    print("\n" + agg["_report.md"], flush=True)
    return out


@app.local_entrypoint()
def run(domain: str = "ood_indian_legal", epochs: int = 3, prep_gen_batch: int = 32,
        bench_limit: int = 150, skip_prep: bool = False):
    """Robust launcher — orchestration happens server-side (see `orchestrate`).
    For long runs launch detached so a client/network drop can't kill it:
        modal run --detach experiments/01-single-domain-dflash/pipeline.py::run --domain ood_indian_legal
    Results always land on the volume under /work/results/<domain>/ regardless."""
    out = orchestrate.remote(domain, epochs=epochs, prep_gen_batch=prep_gen_batch,
                             bench_limit=bench_limit, skip_prep=skip_prep)
    print(out.get("report", out))


@app.local_entrypoint()
def agg_only(domain: str = "ood_indian_legal"):
    print(aggregate.remote(domain)["_report.md"])


@app.local_entrypoint()
def smoke(domain: str = "ood_indian_legal"):
    """Cheap validation of the generalized paths on a real domain before the full
    run: tiny prep, 1-epoch train (lora+full), 3-prompt bench each, aggregate."""
    import json

    print(f"=== SMOKE prep [{domain}] (8 prompts) ===")
    print(json.dumps(prep.remote(domain, max_new_tokens=96, gen_batch=8, limit=8), indent=2))
    print("=== SMOKE train lora + full (1 epoch) ===")
    lc = train_variant.spawn(domain, "lora", epochs=1, val_every=1000)
    fc = train_variant.spawn(domain, "full", epochs=1, val_every=1000)
    print("[lora]", lc.get().get("saved"), " [full]", fc.get().get("saved"))
    print("=== SMOKE bench (3 each) ===")
    bc = {v: bench_variant.spawn(domain, v, max_new_tokens=64, limit=3, warmup=1)
          for v in ("base_dflash", "full_finetune", "lora")}
    for v, c in bc.items():
        print(v, json.dumps(c.get(), indent=2))
    print("=== SMOKE aggregate ===")
    print(aggregate.remote(domain)["_report.md"])
