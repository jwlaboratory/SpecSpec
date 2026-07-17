#!/usr/bin/env python3
"""MoLA for EAGLE3 — the multilingual LoRA specialization experiment, replicated
on the EAGLE3 speculator (RedHatAI/Qwen3-8B-speculator.eagle3).

Same design as ../multilingual/pipeline_langs.py (DFlash), second architecture:
train SIX rank-16 LoRAs on the EAGLE3 draft head — one per language (polish /
korean / italian / japanese / german) + one combined — then bench the 5x3 matrix
(base vs own vs combined) per language. Cross-speculator question: does
specialization still help when the base speculator is stronger?

How this one works:
  * head  — loaded via the `speculators` package (vllm-project), whose
    Eagle3DraftModel ships the CANONICAL EAGLE3 training forward: TTT unrolling
    (3 steps, matching inference) with soft distillation against the verifier's
    own lm_head. We train with that — no custom loss. The package declares the
    aux layers for this head: target layers [2, 18, 33]; a tiny probe still
    settles the hidden_states-tuple offset (0 vs 1) empirically.
  * data  — packed rows (input_ids + document_ids + loss_mask), REUSING the
    DFlash multilingual prep (target-generated token ids) from the volume:
    perfectly apples-to-apples across speculators. Target aux/last hidden
    states are computed live per step.
  * LoRA  — rank-16 on the head's q/k/v/o (q/k/v consume the 8192-dim
    [embed ; fused-hidden] concat). For benching, each adapter is MERGED into
    the head (identical math) and saved as a speculators-format dir.
  * bench — vLLM speculative decoding (same recipe as the repo's
    benchmark_vllm.py), acceptance from vllm:spec_decode_* counter diffs.

Run:
    modal run finetuning/multilingual_eagle/pipeline_eagle.py::smoke
    modal run --detach finetuning/multilingual_eagle/pipeline_eagle.py::run
"""
import pathlib

import modal

LOCAL = pathlib.Path(__file__).resolve().parent            # finetuning/multilingual_eagle/
PARENT = LOCAL.parent                                      # finetuning/
LORA = PARENT / "LoRA" / "lora.py"
LANG_DATA = PARENT / "multilingual" / "data"               # test.jsonl prompts for bench

TARGET_MODEL = "Qwen/Qwen3-8B"
EAGLE_MODEL = "RedHatAI/Qwen3-8B-speculator.eagle3"
GPU = "H200"

LANGS = ["polish", "korean", "italian", "japanese", "german"]
VARIANTS = ["base", "own", "combined"]
NUM_SPEC_TOKENS = 3     # matches the repo's eagle3 domain benchmarks
AUX_LAYERS = (2, 18, 33)  # declared by the speculators pkg for this head
TTT_STEPS = 3
PACK_LEN = 2048

app = modal.App("eagle3-multilingual-lora")

# train image: HF stack + the speculators pkg (canonical Eagle3 classes + loss)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("speculators")  # first, so our exact pins below win any conflicts
    .pip_install("torch==2.9.1", "transformers==4.57.3", "accelerate>=1.0.0",
                 "numpy", "safetensors")
    # speculators pulls a torchvision built against a different torch; after the
    # torch pin its ops fail to register and break transformers' lazy imports
    # ("torchvision::nms does not exist"). transformers doesn't need it — drop it.
    .run_commands("pip uninstall -y torchvision || true")
    .env({"HF_HOME": "/cache", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_file(str(LORA), "/root/lora.py")
    .add_local_dir(str(LANG_DATA), "/root/data/langs")
)

# bench image: vLLM nightly — same recipe as the repo's benchmark_vllm runner
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
)

hf_cache = modal.Volume.from_name("dflash-hf-cache", create_if_missing=True)
work = modal.Volume.from_name("code-sql-pipeline", create_if_missing=True)
VOLS = {"/cache": hf_cache, "/work": work}

PREP = "/work/prep/multilingual"                 # REUSED from the DFlash experiment
MODELS = "/work/models/multilingual_eagle"       # merged speculators-format dirs
RESULTS = "/work/results/multilingual_eagle"


def _read_prompts(lang, split):
    import json
    out = []
    with open(f"/root/data/langs/lang_{lang}/{split}.jsonl", encoding="utf-8") as f:
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


# --------------------------------------------------------------------------- #
# train-side helpers
# --------------------------------------------------------------------------- #
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


def _load_head():
    import torch
    from speculators import SpeculatorModel
    head = SpeculatorModel.from_pretrained(EAGLE_MODEL)
    head = head.to("cuda").to(torch.bfloat16)
    # HF `output_hidden_states` returns the LAST entry post-final-norm, but the
    # speculators TTT loss applies verifier_norm again (double-norm -> distorted
    # soft targets). Our verifier states are already normed — make it a no-op.
    if hasattr(head, "verifier_norm"):
        head.verifier_norm = torch.nn.Identity()
    return head


def _load_prep(langs, split):
    import torch
    data = []
    for lang in langs:
        data += torch.load(f"{PREP}/lang_{lang}/{split}.pt")
    return data


def _pack(records, pack_len=PACK_LEN):
    """Pack records into single rows for the speculators TTT forward:
    (input_ids [1,T], document_ids [1,T], loss_mask [1,T]).

    loss_mask[t] = 1 iff position t+1 is a response token of the SAME document —
    the head's soft target at t is the verifier's next-token distribution."""
    import torch

    packs = []
    cur_ids, cur_doc, cur_mask = [], [], []
    doc = 0

    def flush():
        nonlocal cur_ids, cur_doc, cur_mask, doc
        if cur_ids:
            # pad to EXACTLY pack_len: the speculators flex-attention block mask is
            # built in 128-token blocks and requires q_len to match it exactly.
            # Padding forms its own document (isolated by the doc-mask) with
            # loss_mask=0, so it contributes nothing.
            pad = pack_len - len(cur_ids)
            if pad > 0:
                cur_ids += [0] * pad
                cur_doc += [doc] * pad
                cur_mask += [0] * pad
            packs.append({
                "input_ids": torch.tensor([cur_ids], dtype=torch.long),
                "document_ids": torch.tensor([cur_doc], dtype=torch.long),
                "loss_mask": torch.tensor([cur_mask], dtype=torch.bool),
            })
        cur_ids, cur_doc, cur_mask, doc = [], [], [], 0

    for r in records:
        seq = r["input_ids"].tolist()[:pack_len]
        L = len(seq)
        if L < 8:
            continue
        pl = min(r["prompt_len"], L)
        if len(cur_ids) + L > pack_len:
            flush()
        # loss on positions t where t+1 is a response token (within this doc)
        mask = [1 if (t + 1 >= pl and t + 1 < L) else 0 for t in range(L)]
        cur_ids += seq
        cur_doc += [doc] * L
        cur_mask += mask
        doc += 1
    flush()
    return packs


def _target_states(target, input_ids, offset, order="std"):
    """One target pass -> (aux concat at AUX_LAYERS [1,T,3H], last hidden [1,T,H]).
    order: "std" = [2,18,33] concat order, "rev" = reversed (probe candidate)."""
    import torch
    with torch.no_grad():
        pos = torch.arange(input_ids.shape[1], device="cuda").unsqueeze(0)
        hs = target(input_ids=input_ids, position_ids=pos, use_cache=False,
                    output_hidden_states=True).hidden_states
        layers = list(AUX_LAYERS) if order == "std" else list(AUX_LAYERS)[::-1]
        aux = torch.cat([hs[i + offset] for i in layers], dim=-1).clone()
        return aux, hs[-1].clone()


def _ttt_forward(head, target, pack, offset, order="std"):
    """Run the speculators canonical TTT training forward on one packed row.
    Returns (loss, metrics)."""
    input_ids = pack["input_ids"].to("cuda")
    aux, last_h = _target_states(target, input_ids, offset, order)
    _, loss, metrics = head(
        hidden_states=aux,
        input_ids=input_ids,
        document_ids=pack["document_ids"].to("cuda"),
        loss_mask=pack["loss_mask"].to("cuda"),
        verifier_last_hidden_states=last_h,
        ttt_steps=TTT_STEPS,
    )
    return loss, metrics


def _metric_acc(metrics):
    """Pull a step-0 top-1 accuracy-ish scalar out of the metrics dict if present."""
    for k in sorted(metrics):
        lk = k.lower()
        if "acc" in lk or "top1" in lk or "correct" in lk:
            try:
                return float(metrics[k])
            except Exception:
                continue
    return None


# --------------------------------------------------------------------------- #
# 0) PROBE — settle the hidden_states-tuple offset for the aux layers
# --------------------------------------------------------------------------- #
@app.function(gpu=GPU, image=image, timeout=3600, volumes=VOLS)
def probe(n_packs: int = 3):
    """Find the feature convention the pretrained head was trained on: try
    (offset, concat-order) combos and report base TTT loss + step-0 accuracy.
    The RIGHT convention shows a dramatically lower loss than the rest."""
    import json

    import torch

    tok, target = _load_target()
    head = _load_head().eval()
    packs = _pack(_load_prep(["polish", "german"], "val"))[:n_packs]

    results = {}
    for offset in (1, 0):
        for order in ("std", "rev"):
            losses, accs = [], []
            with torch.no_grad():
                for p in packs:
                    loss, metrics = _ttt_forward(head, target, p, offset, order)
                    losses.append(float(loss))
                    a = _metric_acc(metrics)
                    if a is not None:
                        accs.append(a)
            key = f"{offset}:{order}"
            results[key] = {"loss": round(sum(losses) / len(losses), 4),
                            "acc": round(sum(accs) / len(accs), 4) if accs else None}
            print(f"[probe] offset={offset} order={order} -> {results[key]}", flush=True)
    best = min(results, key=lambda k: results[k]["loss"])
    out = {"results": results, "best": best, "best_loss": results[best]["loss"],
           "best_acc": results[best].get("acc"), "aux_layers": list(AUX_LAYERS)}
    with open("/work/results/eagle_probe.json", "w") as f:
        json.dump(out, f, indent=2)
    work.commit()
    print(f"[probe] BEST {best} -> {results[best]}", flush=True)
    return out


# --------------------------------------------------------------------------- #
# 1) TRAIN — LoRA on the head's q/k/v/o via the canonical TTT loss
# --------------------------------------------------------------------------- #
@app.function(gpu=GPU, image=image, timeout=2 * 3600, volumes=VOLS)
def train_lora(name: str, langs: list, aux: str = "1:std", epochs: int = 3,
               lr: float = 1e-3, val_every: int = 100):
    import os
    import random
    import shutil
    import sys
    import time

    import torch

    sys.path.insert(0, "/root")
    from lora import LoRALinear, inject_lora, lora_state_dict, lora_trainable_parameters

    off_s, _, order = aux.partition(":")
    offset, order = int(off_s), (order or "std")
    tok, target = _load_target()
    head = _load_head()

    replaced = inject_lora(head, rank=16, alpha=32,
                           target_modules=("q_proj", "k_proj", "v_proj", "o_proj"))
    head.to("cuda", dtype=torch.bfloat16)
    trainable = list(lora_trainable_parameters(head))
    print(f"[train:{name}] {len(replaced)} layers adapted "
          f"({sum(p.numel() for p in trainable):,} params) offset={offset} langs={langs}",
          flush=True)

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    train_packs = _pack(_load_prep(langs, "train"))
    val_packs = _pack(_load_prep(langs, "val"))
    print(f"[train:{name}] {len(train_packs)} train packs / {len(val_packs)} val packs",
          flush=True)

    def validate():
        head.eval()
        ls, accs = [], []
        with torch.no_grad():
            for p in val_packs[:6]:
                loss, metrics = _ttt_forward(head, target, p, offset, order)
                ls.append(float(loss))
                a = _metric_acc(metrics)
                if a is not None:
                    accs.append(a)
        head.train()
        return (round(sum(ls) / len(ls), 4),
                round(sum(accs) / len(accs), 4) if accs else None)

    log = {"name": name, "langs": langs, "offset": offset, "epochs": epochs, "lr": lr}
    log["val_initial"] = validate()
    print(f"[train:{name}] initial val {log['val_initial']}", flush=True)
    head.train()
    step, t0 = 0, time.time()
    rng = random.Random(0)
    for ep in range(epochs):
        order = list(range(len(train_packs)))
        rng.shuffle(order)
        for i in order:
            loss, metrics = _ttt_forward(head, target, train_packs[i], offset, order)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
            step += 1
            if step % 25 == 0:
                a = _metric_acc(metrics)
                print(f"[train:{name}] ep{ep} step{step} loss={float(loss):.4f} "
                      f"acc={a if a is not None else 'n/a'} ({time.time()-t0:.0f}s)",
                      flush=True)
            if step % val_every == 0:
                print(f"[train:{name}]   val step{step} {validate()}", flush=True)
    log["val_final"] = validate()
    print(f"[train:{name}] final val {log['val_final']}", flush=True)

    # ---- save LoRA + MERGED speculators-format dir (for the vLLM bench) ----
    # merge into the PRISTINE hub state dict (the loaded head also carries
    # verifier_lm_head/verifier_norm used only by the training loss — exclude).
    os.makedirs(MODELS, exist_ok=True)
    torch.save(lora_state_dict(head), f"{MODELS}/{name}_lora.pt")

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file
    snap = snapshot_download(EAGLE_MODEL,
                             allow_patterns=["config.json", "eagle3.py",
                                             "generation_config.json",
                                             "model.safetensors"])
    base_sd = load_file(os.path.join(snap, "model.safetensors"))
    merged = dict(base_sd)
    n_merged = 0
    for mod_name, m in head.named_modules():
        if isinstance(m, LoRALinear):
            key = f"{mod_name}.weight"
            assert key in merged, f"no hub tensor for adapted layer {key}"
            delta = m.scaling * (m.B.detach().float() @ m.A.detach().float())
            merged[key] = (merged[key].float() + delta.cpu()).to(base_sd[key].dtype)
            n_merged += 1
    out_dir = f"{MODELS}/{name}"
    os.makedirs(out_dir, exist_ok=True)
    for fn in ("config.json", "eagle3.py", "generation_config.json"):
        src = os.path.join(snap, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out_dir, fn))
    save_file({k: v.contiguous().cpu() for k, v in merged.items()},
              os.path.join(out_dir, "model.safetensors"))
    work.commit()
    log["saved"] = out_dir
    log["merged_layers"] = n_merged
    log["train_seconds"] = round(time.time() - t0, 1)
    print(f"[train:{name}] merged {n_merged} layers -> {out_dir}", flush=True)
    return log


# --------------------------------------------------------------------------- #
# 2) BENCH — vLLM spec decode, acceptance from counter diffs (per prompt)
# --------------------------------------------------------------------------- #
@app.function(gpu=GPU, image=vllm_image, timeout=2 * 3600, volumes=VOLS)
def bench(lang: str, variant: str, max_new_tokens: int = 256, limit: int = 100,
          warmup: int = 2):
    import json
    import os
    import time

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.v1.metrics.reader import Counter

    assert variant in VARIANTS
    spec_model = (EAGLE_MODEL if variant == "base"
                  else f"{MODELS}/{lang if variant == 'own' else 'combined'}")
    llm = LLM(model=TARGET_MODEL, dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=0.85, disable_log_stats=False,
              speculative_config={"method": "eagle3", "model": spec_model,
                                  "num_speculative_tokens": NUM_SPEC_TOKENS})
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)

    NAMES = ("vllm:spec_decode_num_drafts", "vllm:spec_decode_num_draft_tokens",
             "vllm:spec_decode_num_accepted_tokens")

    def snap():
        tot = {n: 0.0 for n in NAMES}
        for m in llm.get_metrics():
            if isinstance(m, Counter) and m.name in tot:
                tot[m.name] += m.value
        return tot

    prompts = _read_prompts(lang, "test")[:limit]
    texts = [_chat_text(tok, p) for p in prompts]

    for t in texts[:warmup]:
        llm.generate([t], sp)

    os.makedirs(RESULTS, exist_ok=True)
    out_path = f"{RESULTS}/{lang}_{variant}.jsonl"
    recs = []
    t_start = time.time()
    with open(out_path, "w") as fout:
        for idx, text in enumerate(texts):
            before = snap()
            t0 = time.perf_counter()
            outs = llm.generate([text], sp)
            dt = time.perf_counter() - t0
            after = snap()
            gen = len(outs[0].outputs[0].token_ids)
            drafts = after[NAMES[0]] - before[NAMES[0]]
            proposed = after[NAMES[1]] - before[NAMES[1]]
            accepted = after[NAMES[2]] - before[NAMES[2]]
            rec = {"category": f"{lang}:{variant}", "lang": lang, "variant": variant,
                   "prompt_idx": idx, "num_generated_tokens": gen,
                   "forward_steps": drafts,
                   "accepted_draft_tokens": accepted, "proposed_draft_tokens": proposed,
                   "acceptance_rate": (accepted / proposed) if proposed else 0.0,
                   "mean_accept_length": (1 + accepted / drafts) if drafts else 0.0,
                   "spec_seconds": dt, "spec_tok_s": gen / dt if dt > 0 else 0.0}
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            recs.append(rec)
            if idx % 20 == 0:
                print(f"[bench:{lang}:{variant}] {idx+1}/{len(texts)} "
                      f"accept={rec['acceptance_rate']*100:.0f}% "
                      f"mean_len={rec['mean_accept_length']:.2f} "
                      f"({time.time()-t_start:.0f}s)", flush=True)
    work.commit()

    n = len(recs)
    tp = sum(r["proposed_draft_tokens"] for r in recs)
    summ = {"lang": lang, "variant": variant, "n": n,
            "acceptance_rate_pooled": (sum(r["accepted_draft_tokens"] for r in recs) / tp) if tp else 0.0,
            "mean_accept_length": sum(r["mean_accept_length"] for r in recs) / n,
            "spec_tok_s": sum(r["spec_tok_s"] for r in recs) / n}
    print(f"[bench:{lang}:{variant}] DONE {summ}", flush=True)
    return summ


# --------------------------------------------------------------------------- #
# aggregate — 5x3 matrix (same shape as the DFlash one)
# --------------------------------------------------------------------------- #
@app.function(image=image, timeout=1200, volumes=VOLS)
def aggregate():
    import json

    work.reload()
    rows = {}
    merged = f"{RESULTS}/multilingual_eagle.jsonl"
    with open(merged, "w") as fout:
        for lang in LANGS:
            for variant in VARIANTS:
                try:
                    recs = []
                    with open(f"{RESULTS}/{lang}_{variant}.jsonl") as f:
                        for line in f:
                            if line.strip():
                                recs.append(json.loads(line)); fout.write(line)
                except FileNotFoundError:
                    continue
                if not recs:
                    continue
                n = len(recs)
                tp = sum(r["proposed_draft_tokens"] for r in recs)
                rows[(lang, variant)] = {
                    "n": n,
                    "accept": (sum(r["accepted_draft_tokens"] for r in recs) / tp) if tp else 0.0,
                    "mean_len": sum(r["mean_accept_length"] for r in recs) / n,
                    "tok_s": sum(r["spec_tok_s"] for r in recs) / n,
                }

    csv = ["lang,variant,n,acceptance_rate_pooled,mean_accept_length,spec_tok_s"]
    for (lang, variant), d in rows.items():
        csv.append(f"{lang},{variant},{d['n']},{d['accept']:.4f},{d['mean_len']:.4f},{d['tok_s']:.2f}")
    csv_text = "\n".join(csv) + "\n"

    md = ["# EAGLE3 multilingual LoRA specialization — base vs own vs combined\n",
          f"Per-language held-out test · target **{TARGET_MODEL}** · speculator "
          f"**{EAGLE_MODEL}** (1-layer EAGLE3 head, {NUM_SPEC_TOKENS} spec tokens/step) "
          "· vLLM · temperature 0 · rank-16 LoRA on the head's q/k/v/o (canonical "
          "speculators TTT loss), merged for serving.\n",
          "| language | base accept | own accept | combined accept | base len | own len | combined len |",
          "|---|--:|--:|--:|--:|--:|--:|"]
    wins_own, wins_comb = 0, 0
    for lang in LANGS:
        b = rows.get((lang, "base")); o = rows.get((lang, "own")); c = rows.get((lang, "combined"))
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
        md.append(f"| {lang} | {acc(b)} | {acc(o, b)} | {acc(c, b)} "
                  f"| {ml(b)} | {ml(o, b)} | {ml(c, b)} |")
    md.append(f"\n**own-LoRA beats base on {wins_own}/{len(LANGS)} languages; "
              f"combined beats base on {wins_comb}/{len(LANGS)}.**\n")
    md_text = "\n".join(md)

    with open(f"{RESULTS}/multilingual_eagle_report.md", "w") as f:
        f.write(md_text)
    with open(f"{RESULTS}/multilingual_eagle_comparison.csv", "w") as f:
        f.write(csv_text)
    work.commit()
    print("\n" + md_text)
    return {"_report.md": md_text, "_comparison.csv": csv_text}


# --------------------------------------------------------------------------- #
# orchestration — server-side, launch detached
# --------------------------------------------------------------------------- #
@app.function(image=image, timeout=10 * 3600, volumes=VOLS)
def orchestrate(epochs: int = 3, bench_limit: int = 100, aux: str = ""):
    out = {}
    if not aux:
        p = probe.remote()
        aux = p["best"]
        out["probe"] = p
        print(f"[orchestrate] probe -> offset {aux} (loss {p['best_loss']})", flush=True)

    jobs = {lang: train_lora.spawn(lang, [lang], aux, epochs=epochs) for lang in LANGS}
    jobs["combined"] = train_lora.spawn("combined", LANGS, aux, epochs=epochs)
    out["train"] = {name: c.get().get("val_final") for name, c in jobs.items()}
    print(f"[orchestrate] train done: {out['train']}", flush=True)

    bjobs = {(lang, v): bench.spawn(lang, v, limit=bench_limit)
             for lang in LANGS for v in VARIANTS}
    out["bench"] = {f"{lang}:{v}": c.get() for (lang, v), c in bjobs.items()}
    print("[orchestrate] bench done", flush=True)

    agg = aggregate.remote()
    out["report"] = agg["_report.md"]
    print("\n" + agg["_report.md"], flush=True)
    return out


@app.local_entrypoint()
def run(epochs: int = 3, bench_limit: int = 100, aux: str = ""):
    out = orchestrate.remote(epochs=epochs, bench_limit=bench_limit, aux=aux)
    print(out.get("report", out))


@app.local_entrypoint()
def launch(epochs: int = 3, bench_limit: int = 100, aux: str = ""):
    """Fire-and-forget: spawn the server-side orchestrator and exit immediately.
    Use with --detach on a flaky network — the client is only needed for ~2s:
        modal run --detach finetuning/multilingual_eagle/pipeline_eagle.py::launch
    Results land on the volume under /work/results/multilingual_eagle/ regardless;
    watch with `modal app logs <app-id>`."""
    call = orchestrate.spawn(epochs=epochs, bench_limit=bench_limit, aux=aux)
    print(f"LAUNCHED orchestrate: {call.object_id}")


@app.function(image=image, timeout=6 * 3600, volumes=VOLS)
def bench_all(bench_limit: int = 100):
    """Re-run ONLY the 15 benches (adapters already trained+merged) + aggregate."""
    bjobs = {(lang, v): bench.spawn(lang, v, limit=bench_limit)
             for lang in LANGS for v in VARIANTS}
    out = {f"{lang}:{v}": c.get() for (lang, v), c in bjobs.items()}
    agg = aggregate.remote()
    print("\n" + agg["_report.md"], flush=True)
    return {"bench": out, "report": agg["_report.md"]}


@app.local_entrypoint()
def launch_bench(bench_limit: int = 100):
    call = bench_all.spawn(bench_limit=bench_limit)
    print(f"LAUNCHED bench_all: {call.object_id}")


@app.local_entrypoint()
def probe_only():
    import json
    print(json.dumps(probe.remote(), indent=2))


@app.local_entrypoint()
def agg_only():
    print(aggregate.remote()["_report.md"])


@app.local_entrypoint()
def smoke():
    """Validate every new path cheaply: probe -> short polish train (+merge) ->
    vLLM bench of base and merged-own (2 prompts) -> aggregate."""
    import json

    print("=== SMOKE probe ===")
    p = probe.remote(n_packs=2)
    print(json.dumps(p, indent=2))
    print("=== SMOKE train polish (1 epoch) ===")
    t = train_lora.remote("polish", ["polish"], p["best"], epochs=1, val_every=1000)
    print("[polish]", t.get("saved"), t.get("val_initial"), "->", t.get("val_final"))
    print("=== SMOKE bench polish base + own (2 prompts) ===")
    bb = bench.spawn("polish", "base", max_new_tokens=64, limit=2, warmup=1)
    bo = bench.spawn("polish", "own", max_new_tokens=64, limit=2, warmup=1)
    print("base:", json.dumps(bb.get(), indent=2))
    print("own :", json.dumps(bo.get(), indent=2))
    print("=== SMOKE aggregate ===")
    print(aggregate.remote()["_report.md"])
