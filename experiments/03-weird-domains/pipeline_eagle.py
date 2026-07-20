#!/usr/bin/env python3
"""Weird-domains LoRA specialization — EAGLE3 head on translation / roleplay / poetry.

The EAGLE3 twin of ./pipeline_dflash.py, sharing its prep (target-generated
answers at /work/prep/weird) so the two speculators train on identical data.
Machinery mirrors ../04-multilingual-eagle/pipeline_eagle.py: the `speculators`
package's canonical TTT training forward with the canonical shift_batch
alignment (embed(x_{t+1}) paired with aux_t — the serving-time pairing; v2
trained unshifted, a train/serve mismatch), doc-masked target passes, rank-16
LoRA on the head's q/k/v/o, merge into speculators-format dirs, vLLM bench via
spec-decode counter diffs.

Run AFTER the DFlash weird pipeline's prep stage has populated /work/prep/weird:
    modal run --detach experiments/03-weird-domains/pipeline_eagle.py::launch
"""
import pathlib

import modal

LOCAL = pathlib.Path(__file__).resolve().parent            # experiments/03-weird-domains/
ROOT = LOCAL.parent.parent                                      # repo root
LORA = ROOT / "lib" / "lora.py"
DATA = LOCAL / "data"

TARGET_MODEL = "Qwen/Qwen3-8B"
EAGLE_MODEL = "RedHatAI/Qwen3-8B-speculator.eagle3"
GPU = "H200"

DOMAINS = ["translation", "roleplay", "poetry"]
VARIANTS = ["base", "own", "combined"]
NUM_SPEC_TOKENS = 3
AUX_LAYERS = (2, 18, 33)
TTT_STEPS = 3
PACK_LEN = 2048

app = modal.App("eagle3-weird-domains")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("speculators")
    .pip_install("torch==2.9.1", "transformers==4.57.3", "accelerate>=1.0.0",
                 "numpy", "safetensors")
    .run_commands("pip uninstall -y torchvision || true")
    .env({"HF_HOME": "/cache", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_file(str(LORA), "/root/lora.py")
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

PREP = "/work/prep/weird"                        # SHARED with the DFlash pipeline
MODELS = "/work/models/weird_eagle"
RESULTS = "/work/results/weird_eagle"


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


def _load_head():
    import torch
    from speculators import SpeculatorModel
    head = SpeculatorModel.from_pretrained(EAGLE_MODEL)
    head = head.to("cuda").to(torch.bfloat16)
    # HF hidden_states[-1] is already post-final-norm; the TTT loss would apply
    # verifier_norm again (double-norm -> distorted soft targets). Neutralize it.
    if hasattr(head, "verifier_norm"):
        head.verifier_norm = torch.nn.Identity()
    return head


def _load_prep(domains, split):
    import torch
    data = []
    for d in domains:
        data += torch.load(f"{PREP}/{d}/{split}.pt")
    return data


def _pack(records, pack_len=PACK_LEN):
    import torch

    packs = []
    cur_ids, cur_doc, cur_mask = [], [], []
    doc = 0

    def flush():
        nonlocal cur_ids, cur_doc, cur_mask, doc
        if cur_ids:
            # pad to EXACTLY pack_len: the speculators flex-attention block mask is
            # built in 128-token blocks and requires q_len to match it exactly.
            # Padding forms its own document (isolated) with loss_mask=0.
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
        mask = [1 if (t + 1 >= pl and t + 1 < L) else 0 for t in range(L)]
        cur_ids += seq
        cur_doc += [doc] * L
        cur_mask += mask
        doc += 1
    flush()
    return packs


def _doc_positions_and_mask(document_ids):
    """Per-document 0-based position_ids [1,T] and a block-causal additive
    attention mask [1,1,T,T] that keeps each packed document independent —
    matching serving, where every prompt is its own sequence."""
    import torch
    doc = document_ids[0].to("cuda")
    T = doc.shape[0]
    idx = torch.arange(T, device="cuda")
    starts = torch.cat([torch.zeros(1, dtype=torch.long, device="cuda"),
                        (doc[1:] != doc[:-1]).nonzero(as_tuple=True)[0] + 1])
    doc_start = starts[torch.searchsorted(starts, idx, right=True) - 1]
    pos = (idx - doc_start).unsqueeze(0)
    allowed = (doc.unsqueeze(1) == doc.unsqueeze(0)) & (idx.unsqueeze(1) >= idx.unsqueeze(0))
    mask = torch.zeros(T, T, dtype=torch.bfloat16, device="cuda")
    mask.masked_fill_(~allowed, torch.finfo(torch.bfloat16).min)
    return pos, mask.unsqueeze(0).unsqueeze(0)


def _target_states(target, input_ids, document_ids, offset, order="std"):
    """One doc-masked target pass -> (aux concat [1,T,3H], last hidden [1,T,H]).
    offset maps aux id v -> HF hidden_states[v + offset]; vLLM serving captures
    hidden_states[v] (offset 0)."""
    import torch
    with torch.no_grad():
        pos, attn = _doc_positions_and_mask(document_ids)
        hs = target(input_ids=input_ids, position_ids=pos, attention_mask=attn,
                    use_cache=False, output_hidden_states=True).hidden_states
        layers = list(AUX_LAYERS) if order == "std" else list(AUX_LAYERS)[::-1]
        aux = torch.cat([hs[i + offset] for i in layers], dim=-1).clone()
        return aux, hs[-1].clone()


def _shift_pack(pack, aux, last_h, pack_len=PACK_LEN):
    """Canonical EAGLE3 alignment (speculators eagle3/data.py::shift_batch),
    applied per packed document: the head pairs embed(x_{t+1}) with aux_t and
    is supervised by the verifier's distribution at t+1 (predicting x_{t+2}) —
    exactly the serving-time pairing. Each document drops one slot; the row is
    re-padded to pack_len as an isolated loss_mask=0 document so the flex
    block mask keeps its exact size."""
    import torch
    docs_cpu = pack["document_ids"][0]
    ids = pack["input_ids"][0].to("cuda")
    docs = docs_cpu.to("cuda")
    mask = pack["loss_mask"][0].to("cuda")
    T = ids.shape[0]
    bounds = [0] + [t for t in range(1, T) if docs_cpu[t] != docs_cpu[t - 1]] + [T]
    s_ids, s_docs, s_mask, s_pos, s_aux, s_last = [], [], [], [], [], []
    for s, e in zip(bounds[:-1], bounds[1:]):
        if e - s < 2:
            continue
        s_ids.append(ids[s + 1:e])
        s_docs.append(docs[s + 1:e])
        s_mask.append(mask[s + 1:e])
        s_pos.append(torch.arange(1, e - s, dtype=torch.long, device="cuda"))
        s_aux.append(aux[0, s:e - 1])
        s_last.append(last_h[0, s + 1:e])
    ids2, docs2, mask2 = torch.cat(s_ids), torch.cat(s_docs), torch.cat(s_mask)
    pos2, aux2, last2 = torch.cat(s_pos), torch.cat(s_aux), torch.cat(s_last)
    pad = pack_len - ids2.shape[0]
    if pad > 0:
        pad_doc = int(docs2.max().item()) + 1
        ids2 = torch.cat([ids2, ids2.new_zeros(pad)])
        docs2 = torch.cat([docs2, docs2.new_full((pad,), pad_doc)])
        mask2 = torch.cat([mask2, mask2.new_zeros(pad)])
        pos2 = torch.cat([pos2, pos2.new_ones(pad)])
        aux2 = torch.cat([aux2, aux2.new_zeros(pad, aux2.shape[-1])])
        last2 = torch.cat([last2, last2.new_zeros(pad, last2.shape[-1])])
    return (ids2.unsqueeze(0), docs2.unsqueeze(0), mask2.unsqueeze(0),
            pos2.unsqueeze(0), aux2.unsqueeze(0), last2.unsqueeze(0))


def _ttt_forward(head, target, pack, offset, order="std"):
    input_ids = pack["input_ids"].to("cuda")
    aux, last_h = _target_states(target, input_ids, pack["document_ids"], offset, order)
    ids2, docs2, mask2, pos2, aux2, last2 = _shift_pack(pack, aux, last_h)
    _, loss, metrics = head(
        hidden_states=aux2,
        input_ids=ids2,
        document_ids=docs2,
        loss_mask=mask2,
        verifier_last_hidden_states=last2,
        position_ids=pos2,
        ttt_steps=TTT_STEPS,
    )
    return loss, metrics


def _metric_acc(metrics):
    """Step-0 top-1 accuracy vs the verifier's argmax (masked positions):
    full_acc_0_sum / full_acc_0_total. Directly comparable to serving
    position-1 acceptance at temperature 0."""
    try:
        total = float(metrics["full_acc_0_total"])
        return float(metrics["full_acc_0_sum"]) / total if total else None
    except (KeyError, TypeError):
        return None


def _read_probe_offset():
    """Reuse the offset settled by the multilingual EAGLE probe if available."""
    import json
    try:
        with open("/work/results/eagle_probe.json") as f:
            return str(json.load(f)["best"])
    except Exception:
        return "0"


# --------------------------------------------------------------------------- #
# TRAIN
# --------------------------------------------------------------------------- #
@app.function(gpu=GPU, image=image, timeout=2 * 3600, volumes=VOLS)
def train_lora(name: str, domains: list, aux: str = "", epochs: int = 3,
               lr: float = 1e-4, val_every: int = 100):
    import os
    import random
    import shutil
    import sys
    import time

    import torch

    sys.path.insert(0, "/root")
    from lora import LoRALinear, inject_lora, lora_state_dict, lora_trainable_parameters

    conv = aux if aux else _read_probe_offset()          # "offset:order"
    off_s, _, order = conv.partition(":")
    offset, order = int(off_s), (order or "std")
    tok, target = _load_target()
    head = _load_head()

    replaced = inject_lora(head, rank=16, alpha=32,
                           target_modules=("q_proj", "k_proj", "v_proj", "o_proj"))
    head.to("cuda", dtype=torch.bfloat16)
    trainable = list(lora_trainable_parameters(head))
    print(f"[train:{name}] {len(replaced)} layers "
          f"({sum(p.numel() for p in trainable):,} params) offset={offset} domains={domains}",
          flush=True)

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    train_packs = _pack(_load_prep(domains, "train"))
    val_packs = _pack(_load_prep(domains, "val"))
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

    log = {"name": name, "domains": domains, "offset": offset, "epochs": epochs}
    log["val_initial"] = validate()
    print(f"[train:{name}] initial val {log['val_initial']}", flush=True)
    head.train()
    step, t0 = 0, time.time()
    rng = random.Random(0)
    for ep in range(epochs):
        # NOTE: named `perm`, NOT `order` — a previous version clobbered the
        # aux-concat convention string with this shuffle list, silently training
        # every step on REVERSED aux features (list != "std" -> rev branch).
        perm = list(range(len(train_packs)))
        rng.shuffle(perm)
        for i in perm:
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
    print(f"[train:{name}] merged {n_merged} layers -> {out_dir}", flush=True)
    return log


# --------------------------------------------------------------------------- #
# BENCH (vLLM)
# --------------------------------------------------------------------------- #
@app.function(gpu=GPU, image=vllm_image, timeout=2 * 3600, volumes=VOLS)
def bench(domain: str, variant: str, max_new_tokens: int = 256, limit: int = 100,
          warmup: int = 2):
    import json
    import os
    import time

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.v1.metrics.reader import Counter

    assert variant in VARIANTS
    spec_model = (EAGLE_MODEL if variant == "base"
                  else f"{MODELS}/{domain if variant == 'own' else 'combined'}")
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

    prompts = _read_prompts(domain, "test")[:limit]
    texts = [_chat_text(tok, p) for p in prompts]
    for t in texts[:warmup]:
        llm.generate([t], sp)

    os.makedirs(RESULTS, exist_ok=True)
    out_path = f"{RESULTS}/{domain}_{variant}.jsonl"
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
            rec = {"category": f"{domain}:{variant}", "lang": domain, "variant": variant,
                   "prompt_idx": idx, "num_generated_tokens": gen,
                   "forward_steps": drafts,
                   "accepted_draft_tokens": accepted, "proposed_draft_tokens": proposed,
                   "acceptance_rate": (accepted / proposed) if proposed else 0.0,
                   "mean_accept_length": (1 + accepted / drafts) if drafts else 0.0,
                   "spec_seconds": dt, "spec_tok_s": gen / dt if dt > 0 else 0.0}
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            recs.append(rec)
            if idx % 20 == 0:
                print(f"[bench:{domain}:{variant}] {idx+1}/{len(texts)} "
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
# aggregate
# --------------------------------------------------------------------------- #
@app.function(image=image, timeout=1200, volumes=VOLS)
def aggregate():
    import json

    work.reload()
    rows = {}
    merged = f"{RESULTS}/weird_eagle.jsonl"
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

    md = ["# Weird-domains LoRA specialization (EAGLE3) — base vs own vs combined\n",
          "Domains: translation · roleplay · poetry. Target **Qwen/Qwen3-8B** · "
          f"speculator **{EAGLE_MODEL}** ({NUM_SPEC_TOKENS} spec tokens/step) · vLLM "
          "· temperature 0 · rank-16 LoRA (canonical speculators TTT loss), merged.\n",
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
              f"combined beats base on {wins_comb}/{len(DOMAINS)}.**\n")
    md_text = "\n".join(md)

    with open(f"{RESULTS}/weird_eagle_report.md", "w") as f:
        f.write(md_text)
    with open(f"{RESULTS}/weird_eagle_comparison.csv", "w") as f:
        f.write(csv_text)
    work.commit()
    print("\n" + md_text)
    return {"_report.md": md_text, "_comparison.csv": csv_text}


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
@app.function(image=image, timeout=10 * 3600, volumes=VOLS)
def orchestrate(epochs: int = 3, bench_limit: int = 100, aux: str = ""):
    import os
    import time

    # wait for the SHARED prep (written by the DFlash weird pipeline) and, when no
    # aux override is given, for the probe result (written by the multilingual
    # EAGLE run) so training uses the settled hidden-states offset.
    for _ in range(240):
        work.reload()
        prep_ok = all(os.path.exists(f"{PREP}/{d}/train.pt") for d in DOMAINS)
        probe_ok = bool(aux) or os.path.exists("/work/results/eagle_probe.json")
        if prep_ok and probe_ok:
            break
        print(f"[orchestrate] waiting (prep={prep_ok} probe={probe_ok}) ...", flush=True)
        time.sleep(30)
    else:
        raise RuntimeError("shared weird prep / probe never appeared")

    out = {}
    jobs = {d: train_lora.spawn(d, [d], aux, epochs=epochs) for d in DOMAINS}
    jobs["combined"] = train_lora.spawn("combined", DOMAINS, aux, epochs=epochs)
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
def run(epochs: int = 3, bench_limit: int = 100, aux: str = ""):
    out = orchestrate.remote(epochs=epochs, bench_limit=bench_limit, aux=aux)
    print(out.get("report", out))


@app.local_entrypoint()
def launch(epochs: int = 3, bench_limit: int = 100, aux: str = ""):
    """Fire-and-forget spawn (use with --detach; waits server-side for shared prep)."""
    call = orchestrate.spawn(epochs=epochs, bench_limit=bench_limit, aux=aux)
    print(f"LAUNCHED orchestrate: {call.object_id}")


@app.local_entrypoint()
def agg_only():
    print(aggregate.remote()["_report.md"])


@app.function(image=image, timeout=6 * 3600, volumes=VOLS)
def bench_all(bench_limit: int = 100):
    """Re-run ONLY the 9 benches (adapters already trained+merged) + aggregate."""
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
