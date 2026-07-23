#!/usr/bin/env python3
"""Exp1-language: per-language DFlash LoRAs from a frozen hidden-state dataset.

The experiment (WildChat-4.8M, every language with >= 1,100 conversations — 40
languages, see readme.md):
  1. fetch    (CPU): stream the dataset once; 1000 train / 100 val / 100 test
                     first-turn prompts per language (dedup, skip toxic/redacted)
  2. generate (GPU): Qwen3-8B answers train+val greedily (self-distillation)
  3. capture  (GPU): ONE teacher forward per sequence -> frozen shards
                     {"input_ids" int32 [T], "prompt_len", "hidden" bf16 [T, 5*4096]}
                     via extract_context_feature at target_layer_ids [1,9,17,25,33];
                     also dumps the target's lm_head/embed weights (~2.5 GB once)
  4. verify   (GPU): stored hidden == live teacher hidden; shards drive real LoRA
                     steps with NO 8B teacher loaded
  5. train    (GPU): one LoRA per language + one combined LoRA on all 40,
                     streamed shard-by-shard (the combined set is ~880 GB — it
                     never fits in RAM); teacher-free, so cheap GPUs work
  6. bench    (GPU): base vs own-LoRA vs combined-LoRA on each language's 100
                     held-out test prompts with the instrumented spec_generate;
                     vanilla decode is paired IN-CONTAINER on a subset (memory:
                     wallclock-analytic-model — speedup ~= L/(1+c), DFlash-HF
                     c~=0.44, so acceptance predicts wall-clock; the paired
                     subset just anchors base tok/s)
  7. aggregate: pooled acceptance rate, mean accept length, measured + analytic
                     speedup per (language x variant)

Cost model: hidden states are 40 KB/token -> ~22 GB/language -> ~0.9 TB total
on the `exp1-language-hidden` volume. Storage is the bill; GPU stages are
minutes-to-hours each and fan out across containers.

Run:
    modal run new/exp1-language/pipeline.py::smoke                 # tiny 2-language end-to-end
    modal run --detach new/exp1-language/pipeline.py::launch       # the full experiment
    modal run new/exp1-language/pipeline.py::results               # aggregate + save summary locally
"""
import pathlib

import modal

LOCAL = pathlib.Path(__file__).resolve().parent      # new/exp1-language/
ROOT = LOCAL.parent.parent                           # repo root
LORA = ROOT / "lib" / "lora.py"
ONLINE = ROOT / "lib" / "online_dflash.py"
SPEC_PATCH = ROOT / "lib" / "spec_patch.py"

DRAFT_MODEL = "z-lab/Qwen3-8B-DFlash-b16"
TARGET_MODEL = "Qwen/Qwen3-8B"
GPU_PREP = "A100-40GB"       # generate / capture / verify (no 8B at train)
GPU_TRAIN = "H200"           # train: no-truncation full-2048 batches need the room
GPU_BENCH = "H200"           # bench: comparable to exp-08's fitted constants
MAX_MODEL_LEN = 2048
C_DFLASH_HF = 0.44           # exp-08 fitted per-step overhead (memory: wallclock-analytic-model)
VARIANTS = ["base", "own", "combined"]

# Every WildChat-4.8M language with >= 1,100 conversations (readme.md table),
# minus "Nolang" (= no language detected, not a language). 40 languages.
# NOTE: "Yoruba" is largely a language-detector artifact — kept because it meets
# the threshold, but interpret its results with suspicion.
LANGS = [
    "English", "Russian", "Chinese", "French", "Vietnamese", "Yoruba", "Arabic",
    "Indonesian", "Spanish", "Portuguese", "German", "Persian", "Tagalog",
    "Turkish", "Korean", "Italian", "Maori", "Sotho", "Polish", "Latin",
    "Japanese", "Serbian", "Ukrainian", "Malay", "Dutch", "Esperanto",
    "Romanian", "Hungarian", "Swedish", "Somali", "Estonian", "Tswana",
    "Bulgarian", "Finnish", "Catalan", "Bokmal", "Hebrew", "Welsh", "Hindi",
    "Nynorsk",
]

app = modal.App("exp1-language")

hf_cache = modal.Volume.from_name("dflash-hf-cache", create_if_missing=True)
# Dedicated volume: ~0.9 TB of hidden-state shards, deletable without touching
# the shared pipeline volume.
data_vol = modal.Volume.from_name("exp1-language-hidden", create_if_missing=True)
VOLS = {"/cache": hf_cache, "/data": data_vol}

PROMPTS = "/data/prompts"     # {lang}/{train,val,test}.jsonl        (stage 1)
IDS = "/data/ids"             # {lang}/{train,val}.pt                (stage 2)
SHARDS = "/data/shards"       # {lang}/{split}/shard_NNNN.pt + manifest.json (stage 3)
HEAD_DUMP = "/data/target_head_embed.pt"
MODELS = "/data/models"
RESULTS = "/data/results"

cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("datasets>=3.0.0", "torch==2.9.1", "numpy")
    .env({"HF_HOME": "/cache"})
)

# train/capture/bench image: same pins the z-lab remote code + OnlineDFlashModel
# were validated on (experiments/02-multilingual-dflash)
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

# generation image: vLLM recipe validated in experiments/02-multilingual-dflash
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
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _read_prompts(lang, split):
    import json
    out = []
    with open(f"{PROMPTS}/{lang}/{split}.jsonl", encoding="utf-8") as f:
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


def _shard_paths(langs, split):
    import json
    import os
    paths = []
    for lang in langs:
        d = f"{SHARDS}/{lang}/{split}"
        with open(f"{d}/manifest.json") as f:
            m = json.load(f)
        paths += [os.path.join(d, s) for s in m["shards"]]
    return paths


def make_batch(records, device="cuda", pad_token_id=0, max_seq_len=0):
    """Collate frozen records -> (input_ids, hidden_states, loss_mask) for
    OnlineDFlashModel.forward. Right-padded; pad rows carry loss_mask=0."""
    import torch
    if max_seq_len:
        records = [{**r, "input_ids": r["input_ids"][:max_seq_len],
                    "hidden": r["hidden"][:max_seq_len]} for r in records]
    B = len(records)
    S = max(r["input_ids"].shape[0] for r in records)
    n_ctx = records[0]["hidden"].shape[1]
    input_ids = torch.full((B, S), pad_token_id, dtype=torch.long)
    hidden = torch.zeros((B, S, n_ctx), dtype=torch.bfloat16)
    loss_mask = torch.zeros((B, S), dtype=torch.float)
    for i, r in enumerate(records):
        L = r["input_ids"].shape[0]
        input_ids[i, :L] = r["input_ids"].long()
        hidden[i, :L] = r["hidden"]
        loss_mask[i, min(r["prompt_len"], L):L] = 1.0
    return input_ids.to(device), hidden.to(device), loss_mask.to(device)


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


def _load_head_embed():
    """The target pieces OnlineDFlashModel needs, WITHOUT loading the 8B."""
    import torch
    dump = torch.load(HEAD_DUMP, map_location="cpu")
    V, H = dump["lm_head.weight"].shape
    lm_head = torch.nn.Linear(H, V, bias=False, dtype=torch.bfloat16, device="cuda")
    lm_head.weight.data.copy_(dump["lm_head.weight"])
    embed = torch.nn.Embedding(V, H, dtype=torch.bfloat16, device="cuda")
    embed.weight.data.copy_(dump["embed_tokens.weight"])
    for p in list(lm_head.parameters()) + list(embed.parameters()):
        p.requires_grad_(False)
    return lm_head, embed


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


# --------------------------------------------------------------------------- #
# 1) FETCH — scan the 86 WildChat-4.8M parquet shards IN PARALLEL (one container
#            each, predicate pushdown on the language column), then merge
# --------------------------------------------------------------------------- #
@app.function(image=cpu_image, timeout=3600, volumes=VOLS, max_containers=45,
              memory=8192)
def fetch_shard(fname: str, langs: list, per_lang: int,
                max_prompt_chars: int = 4000):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("allenai/WildChat-4.8M", fname, repo_type="dataset")
    tbl = pq.read_table(path, columns=["language", "conversation",
                                       "toxic", "redacted"])
    mask = pc.and_(pc.is_in(tbl["language"], value_set=pa.array(langs)),
                   pc.and_(pc.invert(tbl["toxic"]), pc.invert(tbl["redacted"])))
    tbl = tbl.filter(mask)
    tbl = tbl.filter(pc.greater(pc.list_value_length(tbl["conversation"]), 0))
    first = pc.list_element(tbl["conversation"], 0)
    roles = pc.struct_field(first, "role").to_pylist()
    contents = pc.struct_field(first, "content").to_pylist()
    lang_col = tbl["language"].to_pylist()

    out = {l: [] for l in langs}
    for lang, role, content in zip(lang_col, roles, contents):
        if len(out[lang]) >= per_lang or role != "user":
            continue
        prompt = (content or "").strip()
        if prompt and len(prompt) <= max_prompt_chars:
            out[lang].append(prompt)
    return {l: v for l, v in out.items() if v}


@app.function(image=cpu_image, timeout=2 * 3600, volumes=VOLS)
def fetch(langs: list, n_train: int = 1000, n_val: int = 100, n_test: int = 100,
          max_prompt_chars: int = 4000, max_files: int = 0):
    import hashlib
    import json
    import os

    from huggingface_hub import HfApi

    files = sorted(f for f in HfApi().list_repo_files(
        "allenai/WildChat-4.8M", repo_type="dataset") if f.endswith(".parquet"))
    if max_files:
        files = files[:max_files]
    need = n_train + n_val + n_test
    print(f"[fetch] scanning {len(files)} parquet shards in parallel", flush=True)

    out = {l: [] for l in langs}
    seen = {l: set() for l in langs}
    # order_outputs=True (default) -> deterministic merge in shard order
    for part in fetch_shard.map(files, kwargs={
            "langs": langs, "per_lang": need,
            "max_prompt_chars": max_prompt_chars}):
        for lang, prompts in part.items():
            for p in prompts:
                if len(out[lang]) >= need:
                    break
                h = hashlib.sha1(p.encode()).hexdigest()
                if h not in seen[lang]:
                    seen[lang].add(h)
                    out[lang].append(p)

    summary = {"files": len(files)}
    for lang, prompts in out.items():
        n = len(prompts)
        if n >= need:
            splits = {"train": prompts[:n_train],
                      "val": prompts[n_train:n_train + n_val],
                      "test": prompts[n_train + n_val:need]}
        else:
            # tail language came up short after filters: split proportionally
            # (10:1:1), never silently drop the language
            v = max(n // 12, 2)
            splits = {"train": prompts[: n - 2 * v],
                      "val": prompts[n - 2 * v: n - v],
                      "test": prompts[n - v:]}
        os.makedirs(f"{PROMPTS}/{lang}", exist_ok=True)
        for split, rows in splits.items():
            with open(f"{PROMPTS}/{lang}/{split}.jsonl", "w", encoding="utf-8") as f:
                for p in rows:
                    f.write(json.dumps({"prompt": p}, ensure_ascii=False) + "\n")
        summary[lang] = {s: len(r) for s, r in splits.items()}
        print(f"[fetch:{lang}] {summary[lang]}", flush=True)
    data_vol.commit()
    return summary


# --------------------------------------------------------------------------- #
# 2) GENERATE — Qwen3-8B answers train+val greedily; save token ids
#               (test stays prompts-only: bench generates it live)
# --------------------------------------------------------------------------- #
@app.function(gpu=GPU_PREP, image=vllm_image, timeout=6 * 3600, volumes=VOLS,
              max_containers=20, retries=2)
def generate(langs: list, max_new_tokens: int = 256, limit: int = 0):
    import json
    import os
    import time

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    llm = LLM(model=TARGET_MODEL, dtype="bfloat16", gpu_memory_utilization=0.90,
              max_model_len=MAX_MODEL_LEN, enable_prefix_caching=True)
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    summary = {}
    for lang in langs:
        os.makedirs(f"{IDS}/{lang}", exist_ok=True)
        for split in ("train", "val"):
            prompts = _read_prompts(lang, split)
            if limit:
                prompts = prompts[:limit]
            texts = [t for t in (_chat_text(tok, p) for p in prompts)
                     if len(tok(t).input_ids) <= MAX_MODEL_LEN - max_new_tokens]
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
            torch.save(records, f"{IDS}/{lang}/{split}.pt")
            resp = [len(r["input_ids"]) - r["prompt_len"] for r in records]
            summary[f"{lang}/{split}"] = {
                "n": len(records), "sec": round(time.time() - t0, 1),
                "mean_resp_len": round(sum(resp) / max(len(resp), 1), 1)}
            print(f"[generate:{lang}:{split}] {summary[f'{lang}/{split}']}", flush=True)
        data_vol.commit()
    return summary


# --------------------------------------------------------------------------- #
# 3) CAPTURE — one frozen teacher forward per sequence; shard hidden states
# --------------------------------------------------------------------------- #
@app.function(gpu=GPU_PREP, image=image, timeout=8 * 3600, volumes=VOLS,
              max_containers=20, memory=16384, retries=2)
def capture(langs: list, batch_tokens: int = 16384, shard_gb: float = 1.5):
    import json
    import os
    import sys
    import time

    import torch
    from transformers import AutoModel

    tok, target = _load_target()

    # The z-lab remote code owns the layer-id -> feature mapping; borrowing its
    # extract fn avoids any hand-rolled offset bug.
    draft = AutoModel.from_pretrained(DRAFT_MODEL, trust_remote_code=True,
                                      dtype=torch.bfloat16)
    extract = sys.modules[type(draft).__module__].extract_context_feature
    target_layer_ids = list(draft.target_layer_ids)
    del draft

    # One-time dump of the target pieces training needs (lm_head + embeddings,
    # ~2.5 GB): with these on the volume, LoRA training never loads the 8B again.
    if not os.path.exists(HEAD_DUMP):
        torch.save({
            "lm_head.weight": target.lm_head.weight.detach().cpu(),
            "embed_tokens.weight": target.get_input_embeddings().weight.detach().cpu(),
            "target_model": TARGET_MODEL,
        }, HEAD_DUMP)
        data_vol.commit()
        print("[capture] saved target lm_head + embed_tokens", flush=True)

    @torch.no_grad()
    def hidden_for(batch_records):
        # right-pad; causal attention means valid positions never see the pad
        # tail — same recipe as experiments/05-interference-ladder/pipeline.py.
        # Call the INNER base model (target.model): the CausalLM wrapper would
        # also compute lm_head logits — B*S x 152k vocab = ~4.6 GB per batch we
        # never use, which OOMs the A100-40GB at batch_tokens=16384.
        B = len(batch_records)
        S = max(r["input_ids"].shape[0] for r in batch_records)
        ids = torch.full((B, S), tok.pad_token_id, dtype=torch.long)
        for i, r in enumerate(batch_records):
            ids[i, :r["input_ids"].shape[0]] = r["input_ids"]
        ids = ids.to("cuda")
        pos = torch.arange(S, device="cuda").unsqueeze(0).expand(B, -1)
        out = target.model(input_ids=ids, position_ids=pos, use_cache=False,
                           output_hidden_states=True)
        return extract(out.hidden_states, target_layer_ids)  # [B, S, 5H] bf16

    summary = {}
    for lang in langs:
        for split in ("train", "val"):
            out_dir = f"{SHARDS}/{lang}/{split}"
            if os.path.exists(f"{out_dir}/manifest.json"):
                summary[f"{lang}/{split}"] = "already captured (manifest exists)"
                print(f"[capture:{lang}:{split}] skip — manifest exists", flush=True)
                continue
            records = torch.load(f"{IDS}/{lang}/{split}.pt", map_location="cpu")
            records = [r for r in records if r["input_ids"].shape[0] <= MAX_MODEL_LEN]
            records.sort(key=lambda r: r["input_ids"].shape[0], reverse=True)
            os.makedirs(out_dir, exist_ok=True)
            shard, shard_bytes, shard_names = [], 0, []
            n_tokens = total_bytes = 0
            shard_limit = int(shard_gb * (1 << 30))
            t0 = time.time()

            def flush():
                nonlocal shard, shard_bytes
                if not shard:
                    return
                name = f"shard_{len(shard_names):04d}.pt"
                torch.save(shard, os.path.join(out_dir, name))
                shard_names.append(name)
                shard, shard_bytes = [], 0

            i = 0
            while i < len(records):
                maxT = records[i]["input_ids"].shape[0]
                B = max(1, batch_tokens // maxT)
                batch = records[i:i + B]
                i += len(batch)
                hid = hidden_for(batch)
                for j, r in enumerate(batch):
                    L = r["input_ids"].shape[0]
                    rec = {"input_ids": r["input_ids"].to(torch.int32),
                           "prompt_len": int(r["prompt_len"]),
                           "hidden": hid[j, :L].to(torch.bfloat16).cpu().clone()}
                    nbytes = rec["hidden"].numel() * 2 + L * 4
                    shard.append(rec)
                    shard_bytes += nbytes
                    n_tokens += L
                    total_bytes += nbytes
                    if shard_bytes >= shard_limit:
                        flush()
            flush()

            manifest = {
                "lang": lang, "split": split, "n": len(records),
                "tokens": n_tokens, "bytes": total_bytes, "shards": shard_names,
                "target_model": TARGET_MODEL, "draft_model": DRAFT_MODEL,
                "target_layer_ids": target_layer_ids, "dtype": "bfloat16",
                "hidden_dim": len(target_layer_ids) * target.config.hidden_size,
            }
            with open(f"{out_dir}/manifest.json", "w") as f:
                json.dump(manifest, f, indent=1)
            summary[f"{lang}/{split}"] = {
                "n": len(records), "tokens": n_tokens,
                "gb": round(total_bytes / (1 << 30), 2),
                "shards": len(shard_names), "sec": round(time.time() - t0, 1)}
            print(f"[capture:{lang}:{split}] {summary[f'{lang}/{split}']}", flush=True)
            data_vol.commit()
    return summary


# --------------------------------------------------------------------------- #
# 4) VERIFY — stored hidden == live teacher hidden; shards drive real LoRA
#             steps WITHOUT the 8B teacher in memory
# --------------------------------------------------------------------------- #
@app.function(gpu=GPU_PREP, image=image, timeout=3600, volumes=VOLS)
def verify(lang: str, split: str = "val", n_check: int = 4, lora_steps: int = 5):
    import sys

    import torch

    sys.path.insert(0, "/root")
    from lora import inject_lora, lora_trainable_parameters
    from online_dflash import OnlineDFlashModel

    paths = _shard_paths([lang], split)
    records = torch.load(paths[0], map_location="cpu")
    tok, target = _load_target()
    draft = _load_draft()
    extract = sys.modules[type(draft).__module__].extract_context_feature
    # only records with an anchorable response window (mirrors training's guard;
    # length-sorted shards put prompt-dominated records first, which would
    # otherwise leave _sample_anchor_positions with zero valid anchors)
    bs_blk = draft.block_size
    checks = [r for r in records
              if r["input_ids"].shape[0] - r["prompt_len"] > 2 * bs_blk][:n_check]
    assert checks, "no records with anchorable responses in first shard"

    # [A] parity diagnostics: recompute hidden live at B=1 and compare to the
    # stored shard (which capture computed inside a right-padded batch).
    # bf16 + different batch shapes legitimately perturbs degenerate tokens
    # (BOS/attention sinks with extreme norms), so cosine stats are DIAGNOSTIC;
    # the pass/fail gate is [B]: does the training loss match on stored vs live?
    lives, cos_all = [], []
    rel_frob = 0.0
    with torch.no_grad():
        for r in checks:
            ids = r["input_ids"].long().unsqueeze(0).to("cuda")
            pos = torch.arange(ids.shape[1], device="cuda").unsqueeze(0)
            out = target(input_ids=ids, position_ids=pos, use_cache=False,
                         output_hidden_states=True)
            live = extract(out.hidden_states, list(draft.target_layer_ids))[0]
            lives.append(live.to(torch.bfloat16).cpu().clone())
            stored = r["hidden"].to("cuda")
            cos_all.append(torch.nn.functional.cosine_similarity(
                live.float(), stored.float(), dim=-1).cpu())
            rel_frob = max(rel_frob, ((live.float() - stored.float()).norm()
                                      / stored.float().norm()).item())
    cos = torch.cat(cos_all)
    cos_stats = {"median": cos.median().item(), "p01": cos.quantile(0.01).item(),
                 "min": cos.min().item(),
                 "frac_below_0.99": (cos < 0.99).float().mean().item(),
                 "max_rel_frobenius": rel_frob}
    print(f"[verify:{lang}] cosine stats {cos_stats}", flush=True)
    del target
    torch.cuda.empty_cache()

    # [B] THE gate — loss parity: run the exact training forward twice with
    # identical seeds (same anchors), once on stored hidden, once on live
    # hidden. If the losses match, the frozen dataset is training-equivalent.
    lm_head, embed = _load_head_embed()
    inject_lora(draft, rank=16, alpha=32,
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"))
    draft.to("cuda", torch.bfloat16)
    online = OnlineDFlashModel(
        draft_model=draft, target_lm_head=lm_head, target_embed_tokens=embed,
        mask_token_id=draft.mask_token_id, block_size=draft.block_size,
        attention_backend="sdpa", num_anchors=48, loss_decay_gamma=7.0,
        loss_type="dflash",
    )
    # loss parity: same batch, same seeded anchors, stored vs live hidden
    live_records = [{**r, "hidden": h} for r, h in zip(checks, lives)]
    parity = {}
    with torch.no_grad():
        for tag, recs in (("stored", checks), ("live", live_records)):
            ids, hid, mask = make_batch(recs, pad_token_id=tok.pad_token_id)
            torch.manual_seed(0)
            out = online(input_ids=ids, hidden_states=hid, loss_mask=mask)
            parity[tag] = {"loss": float(out[0].item()),
                           "acc": float(out[1].item())}
    rel = (abs(parity["stored"]["loss"] - parity["live"]["loss"])
           / max(parity["live"]["loss"], 1e-6))
    parity["loss_rel_diff"] = rel
    print(f"[verify:{lang}] loss parity {parity}", flush=True)
    assert rel < 0.05, f"stored hidden is NOT training-equivalent: {parity}"

    # [C] real LoRA steps purely from frozen shards (no 8B in memory)
    opt = torch.optim.AdamW(list(lora_trainable_parameters(draft)), lr=1e-3)
    draft.train()
    losses, accs = [], []
    bs = 4
    usable = [r for r in records
              if min(r["input_ids"].shape[0], 512) - r["prompt_len"] > 2 * bs_blk]
    assert usable, "no records survive 512-token truncation with a response"
    for step in range(lora_steps):
        batch = usable[(step * bs) % max(len(usable) - bs, 1):][:bs]
        ids, hid, mask = make_batch(batch, pad_token_id=tok.pad_token_id,
                                    max_seq_len=512)
        out = online(input_ids=ids, hidden_states=hid, loss_mask=mask)
        loss, acc = out[0], out[1]
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(round(float(loss.item()), 4))
        accs.append(round(float(acc.item()), 4))
    assert all(torch.isfinite(torch.tensor(losses))), f"non-finite loss: {losses}"
    return {"cosine": cos_stats, "loss_parity": parity,
            "lora_losses": losses, "lora_accs": accs}


# --------------------------------------------------------------------------- #
# 5) TRAIN — one LoRA per language + one combined, streamed shard-by-shard,
#            teacher-free (drafter + lm_head/embed dump only)
# --------------------------------------------------------------------------- #
@app.function(gpu=GPU_TRAIN, image=image, timeout=24 * 3600, volumes=VOLS,
              max_containers=41, memory=32768, retries=2)
def train_lora(name: str, langs: list, epochs: int = 3, lr: float = 1e-3,
               batch_size: int = 12, num_anchors: int = 48, max_seq_len: int = 0,
               rank: int = 16, val_every: int = 400, seed: int = 0,
               batch_tokens: int = 16384):
    import os
    import random
    import sys
    import time

    import torch
    from transformers import AutoTokenizer

    sys.path.insert(0, "/root")
    from lora import inject_lora, lora_state_dict, lora_trainable_parameters
    from online_dflash import OnlineDFlashModel

    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    draft = _load_draft()
    replaced = inject_lora(draft, rank=rank, alpha=2 * rank,
                           target_modules=("q_proj", "k_proj", "v_proj", "o_proj"))
    draft.to("cuda", torch.bfloat16)
    trainable = list(lora_trainable_parameters(draft))
    lm_head, embed = _load_head_embed()
    online = OnlineDFlashModel(
        draft_model=draft, target_lm_head=lm_head, target_embed_tokens=embed,
        mask_token_id=draft.mask_token_id, block_size=draft.block_size,
        attention_backend="sdpa", num_anchors=num_anchors,
        loss_decay_gamma=7.0, loss_type="dflash",
    )
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    block_size = draft.block_size
    rng = random.Random(seed)
    print(f"[train:{name}] rank={rank} {len(replaced)} layers, "
          f"{sum(p.numel() for p in trainable):,} LoRA params, "
          f"{len(langs)} langs", flush=True)

    train_paths = _shard_paths(langs, "train")

    def token_batches(records, budget, max_count):
        """Greedy token-budgeted batches over length-sorted records: each batch
        holds sum(len) <= budget (always >= 1 record) and <= max_count records.
        Replaces fixed batch_size so that full-length (untruncated, up to 2048-
        token) sequences don't OOM while short-sequence languages still batch
        wide -- the truncation-free training path."""
        batch, toks = [], 0
        for r in records:
            L = r["input_ids"].shape[0]
            if batch and (toks + L > budget or len(batch) >= max_count):
                yield batch
                batch, toks = [], 0
            batch.append(r)
            toks += L
        if batch:
            yield batch

    # small fixed val sample spread across languages (one shard per language);
    # keep records with an anchorable response (no truncation now, so use the
    # full sequence length -- prompt-heavy records like Hindi's val keep their
    # response instead of being cut to the prompt at max_seq_len). Sort by length
    # so token_batches produces low-padding batches.
    val_records = []
    per = max(60 // len(langs), 2)
    for l in langs:
        cand = torch.load(_shard_paths([l], "val")[0], map_location="cpu")
        cand = [r for r in cand
                if r["input_ids"].shape[0] - r["prompt_len"] > 2 * block_size]
        val_records += cand[:per]
    val_records.sort(key=lambda r: r["input_ids"].shape[0])

    def run_batch(records, train=True):
        ids, hid, mask = make_batch(records, pad_token_id=tok.pad_token_id,
                                    max_seq_len=max_seq_len)
        anchorable = mask[:, : max(mask.shape[1] - block_size, 0) + 1].sum(dim=1)
        keep = anchorable > block_size + 1
        if keep.sum() == 0:
            return None
        ids, hid, mask = ids[keep], hid[keep], mask[keep]
        try:
            out = online(input_ids=ids, hidden_states=hid, loss_mask=mask)
        except ValueError:
            return None
        loss, acc = out[0], out[1]
        if train:
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
        return float(loss.item()), float(acc.item())

    @torch.no_grad()
    def validate():
        online.eval()
        ls, accs = [], []
        for b in token_batches(val_records, batch_tokens, batch_size):
            r = run_batch(b, train=False)
            if r:
                ls.append(r[0]); accs.append(r[1])
        online.train()
        return (sum(ls) / len(ls), sum(accs) / len(accs)) if ls else (None, None)

    log = {"name": name, "langs": langs, "epochs": epochs, "lr": lr,
           "rank": rank, "shards": len(train_paths), "val": []}
    online.train()
    step = 0
    t0 = time.time()
    log["val_initial"] = validate()
    print(f"[train:{name}] initial val {log['val_initial']}", flush=True)
    def shard_stream(paths):
        # prefetch the next shard from the volume while the GPU trains on the
        # current one — the combined run reads ~880 GB/epoch and would otherwise
        # stall on I/O between shards
        import queue
        import threading
        q = queue.Queue(maxsize=2)

        def loader():
            for p in paths:
                q.put(torch.load(p, map_location="cpu"))
            q.put(None)

        threading.Thread(target=loader, daemon=True).start()
        while True:
            recs = q.get()
            if recs is None:
                return
            yield recs

    for ep in range(epochs):
        rng.shuffle(train_paths)
        for recs in shard_stream(list(train_paths)):
            # records inside a shard are length-sorted (capture): token_batches
            # over them gives near-uniform-length, memory-bounded batches; shuffle
            # the batch order so the optimizer doesn't see length-monotone steps
            batches = list(token_batches(recs, batch_tokens, batch_size))
            rng.shuffle(batches)
            for b in batches:
                r = run_batch(b, train=True)
                if r is None:
                    continue
                step += 1
                if step % 50 == 0:
                    print(f"[train:{name}] ep{ep} step{step} loss={r[0]:.4f} "
                          f"acc={r[1]:.4f} ({time.time()-t0:.0f}s)", flush=True)
                if step % val_every == 0:
                    vl, va = validate()
                    log["val"].append({"step": step, "loss": vl, "acc": va})
                    if vl is not None:
                        print(f"[train:{name}]   val step{step} loss={vl:.4f} "
                              f"acc={va:.4f}", flush=True)
            del recs

    log["val_final"] = validate()
    log["steps"] = step
    print(f"[train:{name}] final val {log['val_final']}", flush=True)
    os.makedirs(MODELS, exist_ok=True)
    sfx = f"_r{rank}" if rank != 16 else ""
    path = f"{MODELS}/{name}{sfx}_lora.pt"
    torch.save(lora_state_dict(draft), path)
    data_vol.commit()
    log["saved"] = path
    log["train_seconds"] = round(time.time() - t0, 1)
    # persist the val curve (initial + every val_every steps + final) so the
    # convergence plot can be regenerated without re-running training
    os.makedirs(f"{RESULTS}/train_logs", exist_ok=True)
    with open(f"{RESULTS}/train_logs/{name}{sfx}.json", "w") as f:
        import json as _json
        _json.dump(log, f, indent=1)
    data_vol.commit()
    return log


# --------------------------------------------------------------------------- #
# 6) BENCH — base vs own vs combined on held-out test prompts.
#            Spec runs on every prompt; vanilla decode paired in-container on a
#            subset (base variant only — vanilla is variant-invariant).
# --------------------------------------------------------------------------- #
@app.function(gpu=GPU_BENCH, image=image, timeout=2 * 3600, volumes=VOLS,
              max_containers=40, retries=2)
def bench(lang: str, variant: str, max_new_tokens: int = 256, limit: int = 100,
          vanilla_limit: int = 15, warmup: int = 2, rank: int = 16):
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

    sfx = f"_r{rank}" if rank != 16 else ""
    if variant != "base":
        from lora import inject_lora
        inject_lora(draft, rank=rank, alpha=2 * rank,
                    target_modules=("q_proj", "k_proj", "v_proj", "o_proj"))
        draft.to("cuda", dtype=torch.bfloat16)
        adapter = (f"{MODELS}/{lang}{sfx}_lora.pt" if variant == "own"
                   else f"{MODELS}/combined{sfx}_lora.pt")
        _load_lora_into(draft, torch.load(adapter, map_location="cuda"))
        print(f"[bench:{lang}:{variant}] loaded {adapter}", flush=True)

    draft.spec_generate = make_instrumented_spec_generate(draft)
    block_size = draft.block_size
    stop_ids = [tok.eos_token_id]
    prompts = _read_prompts(lang, "test")[:limit]

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
    out_path = f"{RESULTS}/{lang}_{variant}{sfx}.jsonl"
    recs = []
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
            steps = len(committed)
            gen = sum(committed)
            accepted = sum(c - 1 for c in committed)
            proposed = steps * (block_size - 1)

            rec = {"lang": lang, "variant": variant, "prompt_idx": idx,
                   "num_input_tokens": n_in, "forward_steps": steps,
                   "num_generated_tokens": gen,
                   "accepted_draft_tokens": accepted,
                   "proposed_draft_tokens": proposed,
                   "acceptance_rate": (accepted / proposed) if proposed else 0.0,
                   "mean_accept_length": (gen / steps) if steps else 0.0,
                   "spec_seconds": spec_dt,
                   "spec_tok_s": len(spec_gen) / spec_dt if spec_dt > 0 else 0.0}

            # paired vanilla decode, same container/GPU (memory:
            # wallclock-analytic-model — unpaired baselines are the failure mode)
            if variant == "base" and idx < vanilla_limit:
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
            if (idx + 1) % 20 == 0:
                print(f"[bench:{lang}:{variant}] {idx+1}/{len(prompts)}", flush=True)
    data_vol.commit()
    pooled_acc = (sum(r["accepted_draft_tokens"] for r in recs)
                  / max(sum(r["proposed_draft_tokens"] for r in recs), 1))
    L = (sum(r["num_generated_tokens"] for r in recs)
         / max(sum(r["forward_steps"] for r in recs), 1))
    print(f"[bench:{lang}:{variant}] acc={pooled_acc:.4f} L={L:.2f}", flush=True)
    return {"lang": lang, "variant": variant, "n": len(recs),
            "acceptance_rate": pooled_acc, "mean_accept_length": L}


# --------------------------------------------------------------------------- #
# 7) AGGREGATE — per (language x variant): pooled acceptance, mean accept
#                length, measured + analytic wall-clock speedup
# --------------------------------------------------------------------------- #
@app.function(image=cpu_image, timeout=1800, volumes=VOLS)
def aggregate(langs: list, rank: int = 16):
    import json
    import os

    sfx = f"_r{rank}" if rank != 16 else ""
    summary = {}
    for lang in langs:
        row = {}
        base_tok_s = None
        for variant in VARIANTS:
            path = f"{RESULTS}/{lang}_{variant}{sfx}.jsonl"
            if not os.path.exists(path):
                continue
            recs = [json.loads(l) for l in open(path) if l.strip()]
            if not recs:
                continue
            acc = (sum(r["accepted_draft_tokens"] for r in recs)
                   / max(sum(r["proposed_draft_tokens"] for r in recs), 1))
            L = (sum(r["num_generated_tokens"] for r in recs)
                 / max(sum(r["forward_steps"] for r in recs), 1))
            tok_s = sum(r["spec_tok_s"] for r in recs) / len(recs)
            row[variant] = {"n": len(recs), "acceptance_rate": round(acc, 4),
                            "mean_accept_length": round(L, 3),
                            "spec_tok_s": round(tok_s, 2),
                            "speedup_analytic": round(L / (1 + C_DFLASH_HF), 3)}
            if variant == "base":
                v = [r["base_tok_s"] for r in recs if "base_tok_s" in r]
                base_tok_s = (sum(v) / len(v)) if v else None
        if base_tok_s:
            for variant in row:
                row[variant]["speedup_measured"] = round(
                    row[variant]["spec_tok_s"] / base_tok_s, 3)
            row["base_vanilla_tok_s"] = round(base_tok_s, 2)
        summary[lang] = row
    with open(f"{RESULTS}/summary{sfx}.json", "w") as f:
        json.dump(summary, f, indent=1)
    data_vol.commit()
    return summary


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
@app.function(image=cpu_image, timeout=24 * 3600, volumes=VOLS)
def full(langs: list, n_train: int = 1000, n_val: int = 100, n_test: int = 100,
         max_new_tokens: int = 256, epochs: int = 3, bench_limit: int = 100,
         rank: int = 16, skip_data: bool = False, skip_train: bool = False,
         skip_fetch: bool = False, skip_generate: bool = False,
         gen_limit: int = 0):
    out = {}
    # chunks of 2 languages -> 20 concurrent generate/capture containers
    chunks = [langs[i:i + 2] for i in range(0, len(langs), 2)]
    if not skip_data:
        if not skip_fetch:
            out["fetch"] = fetch.remote(langs, n_train=n_train, n_val=n_val,
                                        n_test=n_test)
            print("[full] fetch done", flush=True)
        if not skip_generate:
            list(generate.map(chunks, kwargs={"max_new_tokens": max_new_tokens,
                                              "limit": gen_limit}))
            print("[full] generate done", flush=True)
        list(capture.map(chunks))
        print("[full] capture done", flush=True)
        out["verify"] = verify.remote(langs[0])
        print(f"[full] verify: {out['verify']}", flush=True)

    bench_kwargs = {"limit": bench_limit, "rank": rank}
    bench_handles = []
    if not skip_train:
        # all 41 trainings run concurrently; each language's base+own bench is
        # spawned the moment ITS LoRA lands, instead of waiting for the slowest
        # training (the combined LoRA) to finish
        own = {l: train_lora.spawn(l, [l], epochs=epochs, rank=rank)
               for l in langs}
        combined = train_lora.spawn("combined", langs, epochs=epochs, rank=rank)
        out["train"] = []
        for l, h in own.items():
            out["train"].append(h.get())
            bench_handles += [bench.spawn(l, "base", **bench_kwargs),
                              bench.spawn(l, "own", **bench_kwargs)]
        out["train"].append(combined.get())
        print("[full] train done (individual + combined)", flush=True)
        bench_handles += [bench.spawn(l, "combined", **bench_kwargs)
                          for l in langs]
    else:
        bench_handles = [bench.spawn(l, v, **bench_kwargs)
                         for l in langs for v in VARIANTS]
    out["bench"] = [h.get() for h in bench_handles]
    print("[full] bench done", flush=True)
    out["aggregate"] = aggregate.remote(langs, rank=rank)
    return out


@app.local_entrypoint()
def launch(langs: str = "", n_train: int = 1000, n_val: int = 100,
           n_test: int = 100, epochs: int = 3, bench_limit: int = 100,
           rank: int = 16, skip_data: bool = False, skip_train: bool = False,
           skip_fetch: bool = False, skip_generate: bool = False):
    import json
    lang_list = [l.strip() for l in langs.split(",") if l.strip()] or LANGS
    print(json.dumps(full.remote(
        lang_list, n_train=n_train, n_val=n_val, n_test=n_test, epochs=epochs,
        bench_limit=bench_limit, rank=rank, skip_data=skip_data,
        skip_train=skip_train, skip_fetch=skip_fetch,
        skip_generate=skip_generate), indent=2, default=str))


@app.local_entrypoint()
def results(langs: str = "", rank: int = 16):
    import json
    lang_list = [l.strip() for l in langs.split(",") if l.strip()] or LANGS
    summary = aggregate.remote(lang_list, rank=rank)
    out = LOCAL / "results"
    out.mkdir(exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    print(f"saved -> {out / 'summary.json'}")


@app.local_entrypoint()
def smoke():
    """Tiny 2-language end-to-end through EVERY stage (~30 min, one A100 + H200)."""
    import json
    langs = ["French", "Italian"]
    out = {"fetch": fetch.remote(langs, n_train=8, n_val=4, n_test=3,
                                 max_files=4)}
    out["generate"] = generate.remote(langs, max_new_tokens=64)
    out["capture"] = capture.remote(langs, batch_tokens=4096)
    out["verify"] = verify.remote("French")
    out["train_own"] = train_lora.remote("French", ["French"], epochs=1,
                                         batch_size=4, val_every=10)
    out["train_combined"] = train_lora.remote("combined", langs, epochs=1,
                                              batch_size=4, val_every=10)
    out["bench"] = [bench.remote("French", v, max_new_tokens=64, limit=2,
                                 vanilla_limit=2, warmup=1) for v in VARIANTS]
    out["aggregate"] = aggregate.remote(langs)
    print(json.dumps(out, indent=2, default=str))


@app.local_entrypoint()
def curve(lang: str = "Hindi", epochs: int = 3, val_every: int = 25,
          rank: int = 16):
    """Train ONE language with dense validation logging to produce a convergence
    curve. Writes results/train_logs/{lang}.json locally (val loss+acc at
    val_every-step cadence); feed it to make_charts.py::val_loss."""
    import json
    log = train_lora.remote(lang, [lang], epochs=epochs, rank=rank,
                            val_every=val_every)
    out = LOCAL / "results" / "train_logs"
    out.mkdir(parents=True, exist_ok=True)
    sfx = f"_r{rank}" if rank != 16 else ""
    dest = out / f"{lang}{sfx}.json"
    with open(dest, "w") as f:
        json.dump(log, f, indent=1)
    n = len(log.get("val", []))
    print(f"[curve] {lang}: initial={log.get('val_initial')} "
          f"final={log.get('val_final')} ({n} mid points) -> {dest}")
