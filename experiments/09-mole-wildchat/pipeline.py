#!/usr/bin/env python3
"""MoLE on WildChat — do latent LoRA experts beat one combined LoRA?

Everywhere we compared them so far, a single combined adapter ≈ per-domain
specialists (exp 02/03/05/07: gap −0.2..−0.3pp at worst). This experiment asks
whether specialization pays when the MODEL carves the clusters instead of us:
K LoRA experts on the DFlash drafter, mixed per-sequence by a gate on the
pooled target context feature (the exact 20480-dim latent the drafter already
conditions on, same as router/), trained end-to-end on ALL of the wild pool —
no domain labels anywhere. Emergent specialization is read out afterwards by
crossing the learned gate with the (held-out) human domain labels.

Matrix on the wild test splits (n ≤ 100/domain, 16 eval domains):

    base | own | comb_r16 | comb_r64 | mole(K=8, r=8)

  * own       = per-domain specialist r16, trained on that domain's wild train
                (only for the 8 OWN_DOMAINS with 800 train examples)
  * comb_r16  = one r16 LoRA on the FULL wild pool (~15k ex) — active-capacity
                control (matches every prior combined run)
  * comb_r64  = one r64 LoRA on the full pool — total-param control: K=8 r8
                experts have exactly rank-64 worth of LoRA params
  * mole      = 8 experts r8 α16 + LatentGate (20480→64→8, ~1.3M params),
                switch-style load-balance aux loss (coef 0.01)

Wall-clock: per REPORT/exp-08, no per-prompt HF baselines — acceptance is the
metric and predicted speedup = L/(1+c) with the fitted DFlash-HF c ≈ 0.44.
The aggregate emits both, so the own-vs-combined-vs-mole question is answered
directly in speedup units.

Run (detached — server-side orchestration survives client drops):
    modal run --detach experiments/09-mole-wildchat/pipeline.py::launch
Cheap validation first:
    modal run experiments/09-mole-wildchat/pipeline.py::smoke
"""
import pathlib

import modal

LOCAL = pathlib.Path(__file__).resolve().parent            # experiments/09-mole-wildchat/
ROOT = LOCAL.parent.parent
LORA = ROOT / "lib" / "lora.py"
MOLE = ROOT / "lib" / "mole.py"
SPEC_PATCH = ROOT / "lib" / "spec_patch.py"
ONLINE = ROOT / "lib" / "online_dflash.py"
DATA = ROOT / "data" / "wild"

DRAFT_MODEL = "z-lab/Qwen3-8B-DFlash-b16"
TARGET_MODEL = "Qwen/Qwen3-8B"
GPU = "H200"

# The full wild pool — every domain with at least one train example.
TRAIN_DOMAINS = [
    "code_bash", "code_c", "code_cpp", "code_go", "code_java", "code_javascript",
    "code_kotlin", "code_python", "code_r", "code_rust", "code_sql", "code_typescript",
    "lang_arabic", "lang_chinese", "lang_english", "lang_french", "lang_german",
    "lang_italian", "lang_japanese", "lang_korean", "lang_polish", "lang_portuguese",
    "lang_russian", "lang_spanish", "lang_swahili", "lang_turkish", "lang_vietnamese",
    "ood_ascii_art", "ood_chemistry", "ood_customer_support", "ood_financial",
    "ood_legal", "ood_medical", "ood_poetry", "ood_regex", "ood_shell",
    "task_creative_writing", "task_data_generation", "task_email_writing",
    "task_json_extraction", "task_logic_reasoning", "task_math_reasoning",
    "task_question_answering", "task_roleplay_chat", "task_summarization",
    "task_tabular_data", "task_translation",
]
# Evaluated (test split ≥ ~37): diverse across weak/strong base acceptance.
EVAL_DOMAINS = [
    "lang_chinese", "lang_english", "lang_french", "lang_german", "lang_japanese",
    "lang_polish", "lang_russian", "lang_spanish",
    "code_javascript", "code_python", "ood_financial",
    "task_creative_writing", "task_question_answering", "task_roleplay_chat",
    "task_summarization", "task_translation",
]
# Own specialists: eval domains with a full 800-example wild train split.
OWN_DOMAINS = [
    "lang_chinese", "lang_english", "lang_french", "lang_german", "lang_russian",
    "lang_spanish", "task_creative_writing", "task_question_answering",
    "task_roleplay_chat", "task_summarization",
]
VARIANTS = ["base", "own", "comb_r16", "comb_r64", "mole"]

MOLE_EXPERTS = 8
MOLE_RANK = 8
GATE_HIDDEN = 64
LB_COEF = 0.01
C_DFLASH = 0.44          # fitted per-step overhead, exp-08 (speedup ≈ L/(1+c))

app = modal.App("dflash-mole-wildchat")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.9.1", "transformers==4.57.3", "accelerate>=1.0.0",
                 "datasets>=3.0.0", "numpy")
    .env({"HF_HOME": "/cache", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_file(str(ONLINE), "/root/online_dflash.py")
    .add_local_file(str(LORA), "/root/lora.py")
    .add_local_file(str(MOLE), "/root/mole.py")
    .add_local_file(str(SPEC_PATCH), "/root/spec_patch.py")
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

PREP = "/work/prep/mole_wild"
MODELS = "/work/models/mole_wild"
RESULTS = "/work/results/mole_wild"


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


MAX_PROMPT_TOKENS = 1792   # wild prompts can be huge; 2048 engine ctx − 256 gen


def _filter_prompts(tok, prompts):
    """Drop prompts whose chat-templated form exceeds MAX_PROMPT_TOKENS.
    Applied identically at prep and bench so every variant sees the same set
    (training truncates at 512 tokens anyway — overlong prompts have no loss
    region and would be dropped there too)."""
    kept = []
    for p in prompts:
        if len(tok(_chat_text(tok, p)).input_ids) <= MAX_PROMPT_TOKENS:
            kept.append(p)
    return kept


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
# 1) PREP — target generations for the whole wild pool (train+val)
# --------------------------------------------------------------------------- #
@app.function(gpu=GPU, image=vllm_image, timeout=6 * 3600, volumes=VOLS)
def prep(max_new_tokens: int = 256, limit: int = 0):
    import os
    import time

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    llm = LLM(model=TARGET_MODEL, dtype="bfloat16", gpu_memory_utilization=0.90,
              max_model_len=2048, enable_prefix_caching=True)
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    summary = {}
    for domain in TRAIN_DOMAINS:
        os.makedirs(f"{PREP}/{domain}", exist_ok=True)
        for split in ("train", "val"):
            out_path = f"{PREP}/{domain}/{split}.pt"
            prompts = _filter_prompts(tok, _read_prompts(domain, split))
            if limit:
                prompts = prompts[:limit]
            if not prompts:
                torch.save([], out_path)
                continue
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
            torch.save(records, out_path)
            summary[f"{domain}/{split}"] = {"n": len(records),
                                            "sec": round(time.time() - t0, 1)}
            print(f"[prep:{domain}:{split}] {summary[f'{domain}/{split}']}", flush=True)
        work.commit()
    return summary


# --------------------------------------------------------------------------- #
# shared training scaffolding
# --------------------------------------------------------------------------- #
def _make_batch(records, tok, max_seq_len):
    """Returns (input_ids, loss_mask, prompt_mask) on cuda."""
    import torch
    seqs = [r["input_ids"][:max_seq_len] for r in records]
    S = max(len(s) for s in seqs)
    B = len(seqs)
    input_ids = torch.full((B, S), tok.pad_token_id, dtype=torch.long)
    loss_mask = torch.zeros((B, S), dtype=torch.float)
    prompt_mask = torch.zeros((B, S), dtype=torch.float)
    for i, (s, r) in enumerate(zip(seqs, records)):
        L = len(s)
        input_ids[i, :L] = s
        p = min(r["prompt_len"], L)
        loss_mask[i, p:L] = 1.0
        prompt_mask[i, :p] = 1.0
    return input_ids.to("cuda"), loss_mask.to("cuda"), prompt_mask.to("cuda")


def _load_pool(domains, split):
    import torch
    data = []
    for domain in domains:
        try:
            data += torch.load(f"{PREP}/{domain}/{split}.pt")
        except FileNotFoundError:
            pass
    return data


# --------------------------------------------------------------------------- #
# 2a) TRAIN — plain LoRA (own specialists + combined controls)
# --------------------------------------------------------------------------- #
@app.function(gpu=GPU, image=image, timeout=23 * 3600, volumes=VOLS)
def train_lora(name: str, domains: list, epochs: int = 3, lr: float = 1e-3,
               batch_size: int = 12, num_anchors: int = 48, max_seq_len: int = 512,
               val_every: int = 500, rank: int = 16):
    import sys
    import time

    import torch

    sys.path.insert(0, "/root")
    from lora import inject_lora, lora_state_dict, lora_trainable_parameters
    from online_dflash import OnlineDFlashModel

    tok, target = _load_target()
    draft = _load_draft()
    extract = sys.modules[type(draft).__module__].extract_context_feature
    target_layer_ids = draft.target_layer_ids
    block_size = draft.block_size

    inject_lora(draft, rank=rank, alpha=2 * rank)
    draft.to("cuda", dtype=torch.bfloat16)
    trainable = list(lora_trainable_parameters(draft))
    print(f"[train:{name}] rank={rank} {sum(p.numel() for p in trainable):,} params, "
          f"{len(domains)} domains", flush=True)

    online = OnlineDFlashModel(
        draft_model=draft, target_lm_head=target.lm_head,
        target_embed_tokens=target.get_input_embeddings(),
        mask_token_id=draft.mask_token_id, block_size=block_size,
        attention_backend="sdpa", num_anchors=num_anchors,
        loss_decay_gamma=7.0, loss_type="dflash",
    )
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)

    train_data = _load_pool(domains, "train")
    val_data = _load_pool(domains, "val")
    print(f"[train:{name}] {len(train_data)} train / {len(val_data)} val", flush=True)

    @torch.no_grad()
    def target_hidden(input_ids):
        pos = torch.arange(input_ids.shape[1], device="cuda").unsqueeze(0).expand(input_ids.shape[0], -1)
        out = target(input_ids=input_ids, position_ids=pos, use_cache=False, output_hidden_states=True)
        return extract(out.hidden_states, target_layer_ids).clone()

    def run_batch(records, train=True):
        input_ids, loss_mask, _ = _make_batch(records, tok, max_seq_len)
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
        stride = max(len(val_data) // 60, 1)
        sample = val_data[::stride][:60]
        for i in range(0, len(sample), batch_size):
            r = run_batch(sample[i:i + batch_size], train=False)
            if r:
                ls.append(r[0]); accs.append(r[1])
        online.train()
        return (sum(ls) / len(ls), sum(accs) / len(accs)) if ls else (None, None)

    log = {"name": name, "n_domains": len(domains), "epochs": epochs, "lr": lr,
           "rank": rank, "val": []}
    online.train()
    step = 0
    t0 = time.time()
    log["val_initial"] = validate()
    print(f"[train:{name}] initial val {log['val_initial']}", flush=True)
    for ep in range(epochs):
        order = torch.randperm(len(train_data)).tolist()
        for i in range(0, len(train_data), batch_size):
            r = run_batch([train_data[j] for j in order[i:i + batch_size]], train=True)
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

    import os as _os
    _os.makedirs(MODELS, exist_ok=True)
    path = f"{MODELS}/{name}_lora.pt"
    torch.save(lora_state_dict(draft), path)
    work.commit()
    log["saved"] = path
    log["train_seconds"] = round(time.time() - t0, 1)
    print(f"[train:{name}] saved -> {path}", flush=True)
    return log


# --------------------------------------------------------------------------- #
# 2b) TRAIN — MoLE (experts + latent gate, end-to-end, no labels)
# --------------------------------------------------------------------------- #
@app.function(gpu=GPU, image=image, timeout=23 * 3600, volumes=VOLS)
def train_mole(epochs: int = 3, lr: float = 1e-3, batch_size: int = 12,
               num_anchors: int = 48, max_seq_len: int = 512, val_every: int = 500,
               num_experts: int = MOLE_EXPERTS, rank: int = MOLE_RANK,
               lb_coef: float = LB_COEF):
    import sys
    import time

    import torch

    sys.path.insert(0, "/root")
    from mole import (LatentGate, gate_entropy, inject_mole, load_balance_loss,
                      mole_expert_state_dict, mole_trainable_parameters,
                      pool_prompt_feature)
    from online_dflash import OnlineDFlashModel

    tok, target = _load_target()
    draft = _load_draft()
    extract = sys.modules[type(draft).__module__].extract_context_feature
    target_layer_ids = draft.target_layer_ids
    block_size = draft.block_size
    feat_dim = target.config.hidden_size * len(target_layer_ids)

    replaced, box = inject_mole(draft, num_experts=num_experts, rank=rank,
                                alpha=2 * rank)
    draft.to("cuda", dtype=torch.bfloat16)
    gate = LatentGate(feat_dim=feat_dim, hidden=GATE_HIDDEN,
                      num_experts=num_experts).to("cuda")  # fp32 on purpose
    expert_params = list(mole_trainable_parameters(draft))
    gate_params = list(gate.parameters())
    trainable = expert_params + gate_params
    print(f"[train:mole] K={num_experts} r={rank} {len(replaced)} layers, "
          f"{sum(p.numel() for p in expert_params):,} expert + "
          f"{sum(p.numel() for p in gate_params):,} gate params", flush=True)

    online = OnlineDFlashModel(
        draft_model=draft, target_lm_head=target.lm_head,
        target_embed_tokens=target.get_input_embeddings(),
        mask_token_id=draft.mask_token_id, block_size=block_size,
        attention_backend="sdpa", num_anchors=num_anchors,
        loss_decay_gamma=7.0, loss_type="dflash",
    )
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)

    train_data = _load_pool(TRAIN_DOMAINS, "train")
    val_data = _load_pool(TRAIN_DOMAINS, "val")
    print(f"[train:mole] {len(train_data)} train / {len(val_data)} val (pooled, "
          f"no labels)", flush=True)

    @torch.no_grad()
    def target_hidden(input_ids):
        pos = torch.arange(input_ids.shape[1], device="cuda").unsqueeze(0).expand(input_ids.shape[0], -1)
        out = target(input_ids=input_ids, position_ids=pos, use_cache=False, output_hidden_states=True)
        return extract(out.hidden_states, target_layer_ids).clone()

    def run_batch(records, train=True):
        input_ids, loss_mask, prompt_mask = _make_batch(records, tok, max_seq_len)
        anchorable = loss_mask[:, : max(loss_mask.shape[1] - block_size, 0) + 1].sum(dim=1)
        keep = anchorable > block_size + 1
        if keep.sum() == 0:
            return None
        input_ids, loss_mask, prompt_mask = input_ids[keep], loss_mask[keep], prompt_mask[keep]
        hid = target_hidden(input_ids)
        g = gate(pool_prompt_feature(hid, prompt_mask))
        box.g = g
        try:
            loss, acc, _ = online(input_ids=input_ids, hidden_states=hid, loss_mask=loss_mask)
        except ValueError:
            return None
        lb = load_balance_loss(g)
        total = loss + lb_coef * lb
        if train:
            opt.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
        return (float(loss.item()), float(acc.item()), float(lb.item()),
                float(gate_entropy(g).item()))

    @torch.no_grad()
    def validate():
        online.eval(); gate.eval()
        ls, accs, ents, marg = [], [], [], None
        stride = max(len(val_data) // 60, 1)
        sample = val_data[::stride][:60]
        gs = []
        for i in range(0, len(sample), batch_size):
            r = run_batch(sample[i:i + batch_size], train=False)
            if r:
                ls.append(r[0]); accs.append(r[1]); ents.append(r[3])
                gs.append(box.g.detach().float().cpu())
        online.train(); gate.train()
        if not ls:
            return None
        marg = torch.cat(gs).mean(0)
        return {"loss": sum(ls) / len(ls), "acc": sum(accs) / len(accs),
                "sample_entropy": sum(ents) / len(ents),
                "marginal": [round(x, 4) for x in marg.tolist()]}

    log = {"name": "mole", "num_experts": num_experts, "rank": rank,
           "lb_coef": lb_coef, "epochs": epochs, "lr": lr, "val": []}
    online.train(); gate.train()
    step = 0
    t0 = time.time()
    log["val_initial"] = validate()
    print(f"[train:mole] initial val {log['val_initial']}", flush=True)
    for ep in range(epochs):
        order = torch.randperm(len(train_data)).tolist()
        for i in range(0, len(train_data), batch_size):
            r = run_batch([train_data[j] for j in order[i:i + batch_size]], train=True)
            if r is None:
                continue
            step += 1
            if step % 50 == 0:
                print(f"[train:mole] ep{ep} step{step} loss={r[0]:.4f} acc={r[1]:.4f} "
                      f"lb={r[2]:.3f} H={r[3]:.3f} ({time.time()-t0:.0f}s)", flush=True)
            if step % val_every == 0:
                v = validate()
                log["val"].append({"step": step, **(v or {})})
                print(f"[train:mole]   val step{step} {v}", flush=True)
    log["val_final"] = validate()
    print(f"[train:mole] final val {log['val_final']}", flush=True)

    import os as _os
    _os.makedirs(MODELS, exist_ok=True)
    path = f"{MODELS}/mole_k{num_experts}r{rank}.pt"
    torch.save({"experts": mole_expert_state_dict(draft),
                "gate": gate.state_dict(),
                "config": {"num_experts": num_experts, "rank": rank,
                           "alpha": 2 * rank, "gate_hidden": GATE_HIDDEN,
                           "feat_dim": feat_dim, "lb_coef": lb_coef}},
               path)
    work.commit()
    log["saved"] = path
    log["train_seconds"] = round(time.time() - t0, 1)
    print(f"[train:mole] saved -> {path}", flush=True)
    return log


# --------------------------------------------------------------------------- #
# 3) BENCH — acceptance per domain; mole logs its gate per prompt
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
          warmup: int = 2):
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
    extract = sys.modules[type(draft).__module__].extract_context_feature
    target_layer_ids = draft.target_layer_ids

    box = gate = None
    if variant == "mole":
        from mole import LatentGate, inject_mole, load_mole_expert_state
        ckpt = torch.load(f"{MODELS}/mole_k{MOLE_EXPERTS}r{MOLE_RANK}.pt",
                          map_location="cuda")
        cfg = ckpt["config"]
        _, box = inject_mole(draft, num_experts=cfg["num_experts"],
                             rank=cfg["rank"], alpha=cfg["alpha"])
        draft.to("cuda", dtype=torch.bfloat16)
        load_mole_expert_state(draft, ckpt["experts"])
        gate = LatentGate(feat_dim=cfg["feat_dim"], hidden=cfg["gate_hidden"],
                          num_experts=cfg["num_experts"]).to("cuda")
        gate.load_state_dict(ckpt["gate"])
        gate.eval()
        print(f"[bench:{domain}:mole] loaded K={cfg['num_experts']} r={cfg['rank']}",
              flush=True)
    elif variant != "base":
        from lora import inject_lora
        rank = 64 if variant == "comb_r64" else 16
        inject_lora(draft, rank=rank, alpha=2 * rank)
        draft.to("cuda", dtype=torch.bfloat16)
        adapter = (f"{MODELS}/{domain}_lora.pt" if variant == "own"
                   else f"{MODELS}/{variant}_lora.pt")
        _load_lora_into(draft, torch.load(adapter, map_location="cuda"))
        print(f"[bench:{domain}:{variant}] loaded {adapter}", flush=True)

    draft.spec_generate = make_instrumented_spec_generate(draft)
    block_size = draft.block_size
    stop_ids = [tok.eos_token_id]
    prompts = _filter_prompts(tok, _read_prompts(domain, "test"))[:limit]

    def build_ids(p):
        return tok([_chat_text(tok, p)], return_tensors="pt").input_ids.to("cuda")

    @torch.no_grad()
    def set_gate(input_ids):
        """One target prefill outside the timed region: pooled feature -> g.
        In real serving this reuses the prefill the drafter needs anyway."""
        pos = torch.arange(input_ids.shape[1], device="cuda").unsqueeze(0)
        out = target(input_ids=input_ids, position_ids=pos, use_cache=False,
                     output_hidden_states=True)
        feat = extract(out.hidden_states, target_layer_ids)
        box.g = gate(feat.float().mean(1))
        return box.g[0].float().cpu().tolist()

    warm = build_ids(prompts[0])
    if variant == "mole":
        set_gate(warm)
    for _ in range(warmup):
        with torch.inference_mode():
            draft.spec_generate(target=target, input_ids=warm, max_new_tokens=64,
                                stop_token_ids=stop_ids, temperature=0.0)
    torch.cuda.synchronize()

    os.makedirs(RESULTS, exist_ok=True)
    out_path = f"{RESULTS}/{domain}_{variant}.jsonl"
    recs = []
    t_start = time.time()
    with open(out_path, "w") as fout:
        for idx, prompt in enumerate(prompts):
            input_ids = build_ids(prompt)
            g_list = set_gate(input_ids) if variant == "mole" else None
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.inference_mode():
                out_ids, committed = draft.spec_generate(
                    target=target, input_ids=input_ids, max_new_tokens=max_new_tokens,
                    stop_token_ids=stop_ids, temperature=0.0)
            torch.cuda.synchronize(); spec_dt = time.perf_counter() - t0
            steps = len(committed)
            gen = sum(committed)
            accepted = sum(c - 1 for c in committed)
            proposed = steps * (block_size - 1)
            rec = {"domain": domain, "variant": variant, "prompt_idx": idx,
                   "num_input_tokens": input_ids.shape[1],
                   "forward_steps": steps, "num_generated_tokens": gen,
                   "accepted_draft_tokens": accepted, "proposed_draft_tokens": proposed,
                   "acceptance_rate": (accepted / proposed) if proposed else 0.0,
                   "mean_accept_length": (gen / steps) if steps else 0.0,
                   "committed_per_step": committed, "spec_seconds": spec_dt}
            if g_list is not None:
                rec["gate"] = [round(x, 4) for x in g_list]
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
            "mean_accept_length": sum(r["mean_accept_length"] for r in recs) / max(n, 1)}
    print(f"[bench:{domain}:{variant}] DONE {summ}", flush=True)
    return summ


# --------------------------------------------------------------------------- #
# 4) aggregate — 16x5 matrix, predicted speedups, gate-vs-domain readout
# --------------------------------------------------------------------------- #
@app.function(image=image, timeout=1200, volumes=VOLS)
def aggregate():
    import json
    import math

    work.reload()
    rows = {}
    gates = {}   # domain -> list of gate vectors
    merged = f"{RESULTS}/mole_wildchat.jsonl"
    with open(merged, "w") as fout:
        for domain in EVAL_DOMAINS:
            for variant in VARIANTS:
                if variant == "own" and domain not in OWN_DOMAINS:
                    continue
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
                L = sum(r["mean_accept_length"] for r in recs) / n
                rows[(domain, variant)] = {
                    "n": n,
                    "accept": (sum(r["accepted_draft_tokens"] for r in recs) / tp) if tp else 0.0,
                    "mean_len": L,
                    "pred_speedup": L / (1.0 + C_DFLASH),
                }
                if variant == "mole":
                    gates[domain] = [r["gate"] for r in recs if "gate" in r]

    csv = ["domain,variant,n,acceptance_rate_pooled,mean_accept_length,pred_speedup"]
    for (domain, variant), d in rows.items():
        csv.append(f"{domain},{variant},{d['n']},{d['accept']:.4f},"
                   f"{d['mean_len']:.4f},{d['pred_speedup']:.4f}")
    csv_text = "\n".join(csv) + "\n"

    md = ["# MoLE on WildChat — base vs own vs comb_r16 vs comb_r64 vs mole\n",
          "Wild test splits (n ≤ 100/domain) · target **Qwen/Qwen3-8B** · drafter "
          "**z-lab/Qwen3-8B-DFlash-b16** · temperature 0.\n",
          f"mole = K={MOLE_EXPERTS} experts r={MOLE_RANK} (rank-64-equivalent params) "
          "+ latent gate on the pooled 20480-dim context feature, trained on the "
          "FULL unlabeled wild pool. own = per-domain r16 specialist (800 wild ex). "
          "comb_r16/comb_r64 = one LoRA on the full pool.\n",
          f"pred_speedup = mean_accept_length / (1 + c), c = {C_DFLASH} "
          "(exp-08 fitted DFlash-HF); acceptance deltas in parentheses vs base.\n",
          "| domain | base | own | comb_r16 | comb_r64 | mole |",
          "|---|--:|--:|--:|--:|--:|"]

    def cell(d, ref=None):
        if d is None:
            return "—"
        s = f"{d['accept']*100:.1f}%"
        if ref is not None:
            diff = (d["accept"] - ref["accept"]) * 100
            s += f" ({'+' if diff >= 0 else ''}{diff:.1f}pp)"
        return s

    gaps = {v: [] for v in ("own", "comb_r16", "comb_r64", "mole")}
    sp_gaps = {v: [] for v in ("own", "comb_r16", "comb_r64", "mole")}
    for domain in EVAL_DOMAINS:
        b = rows.get((domain, "base"))
        if not b:
            continue
        cells = [cell(b)]
        for v in ("own", "comb_r16", "comb_r64", "mole"):
            d = rows.get((domain, v))
            cells.append(cell(d, b))
            if d:
                gaps[v].append((d["accept"] - b["accept"]) * 100)
                sp_gaps[v].append(d["pred_speedup"] - b["pred_speedup"])
        md.append(f"| {domain} | " + " | ".join(cells) + " |")

    md.append("\n**mean vs base:** " + " · ".join(
        f"{v}: {sum(g)/len(g):+.2f}pp / {sum(s)/len(s):+.3f}x"
        for (v, g), s in zip(gaps.items(), sp_gaps.values()) if g))

    # own-vs-comb and mole-vs-comb head-to-heads in speedup units
    h2h = []
    for a, bvar in (("own", "comb_r16"), ("mole", "comb_r16"), ("mole", "comb_r64")):
        ds = [(rows[(d, a)]["pred_speedup"] - rows[(d, bvar)]["pred_speedup"])
              for d in EVAL_DOMAINS if (d, a) in rows and (d, bvar) in rows]
        if ds:
            h2h.append(f"{a} − {bvar}: {sum(ds)/len(ds):+.4f}x mean "
                       f"(range {min(ds):+.3f}..{max(ds):+.3f})")
    md.append("\n**head-to-head predicted speedup:** " + " · ".join(h2h) + "\n")

    # gate readout: mean mixture per domain + entropies
    if gates:
        K = len(next(iter(gates.values()))[0])
        md.append("\n## Latent gate vs (held-out) domain labels\n")
        md.append("| domain | " + " | ".join(f"e{k}" for k in range(K)) +
                  " | top | H(sample) |")
        md.append("|---|" + "--:|" * K + "--:|--:|")
        marginal = [0.0] * K
        total = 0
        gate_csv = ["domain," + ",".join(f"e{k}" for k in range(K))]
        for domain in EVAL_DOMAINS:
            gs = gates.get(domain)
            if not gs:
                continue
            mean_g = [sum(g[k] for g in gs) / len(gs) for k in range(K)]
            for k in range(K):
                marginal[k] += sum(g[k] for g in gs)
            total += len(gs)
            ent = sum(-sum(p * math.log(p + 1e-9) for p in g) for g in gs) / len(gs)
            top = max(range(K), key=lambda k: mean_g[k])
            md.append(f"| {domain} | " +
                      " | ".join(f"{x:.2f}" for x in mean_g) +
                      f" | e{top} | {ent:.2f} |")
            gate_csv.append(f"{domain}," + ",".join(f"{x:.4f}" for x in mean_g))
        if total:
            marg = [m / total for m in marginal]
            marg_ent = -sum(p * math.log(p + 1e-9) for p in marg)
            md.append(f"\nmarginal usage: {[round(x, 3) for x in marg]} · "
                      f"H(marginal) = {marg_ent:.2f} nats (uniform = "
                      f"{math.log(K):.2f}) — specialization = low H(sample), "
                      f"high H(marginal).\n")
        with open(f"{RESULTS}/gate_by_domain.csv", "w") as f:
            f.write("\n".join(gate_csv) + "\n")

    md_text = "\n".join(md)
    with open(f"{RESULTS}/mole_report.md", "w") as f:
        f.write(md_text)
    with open(f"{RESULTS}/mole_comparison.csv", "w") as f:
        f.write(csv_text)
    work.commit()
    print("\n" + md_text)
    return {"_report.md": md_text, "_comparison.csv": csv_text}


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
@app.function(image=image, timeout=23 * 3600, volumes=VOLS)
def orchestrate(epochs: int = 3, bench_limit: int = 100, skip_prep: bool = False,
                prep_limit: int = 0, bench_max_new_tokens: int = 256,
                skip_train: bool = False):
    out = {}
    if not skip_prep:
        out["prep"] = prep.remote(limit=prep_limit)
        print("[orchestrate] prep done", flush=True)

    bjobs = {(d, "base"): bench.spawn(d, "base", max_new_tokens=bench_max_new_tokens,
                                      limit=bench_limit)
             for d in EVAL_DOMAINS}

    if not skip_train:
        jobs = {"mole": train_mole.spawn(epochs=epochs),
                "comb_r16": train_lora.spawn("comb_r16", TRAIN_DOMAINS,
                                             epochs=epochs, rank=16),
                "comb_r64": train_lora.spawn("comb_r64", TRAIN_DOMAINS,
                                             epochs=epochs, rank=64)}
        for d in OWN_DOMAINS:
            jobs[d] = train_lora.spawn(d, [d], epochs=epochs, rank=16)
        out["train"] = {name: c.get().get("val_final") for name, c in jobs.items()}
        print(f"[orchestrate] train done: {out['train']}", flush=True)

    for d in EVAL_DOMAINS:
        for v in ("own", "comb_r16", "comb_r64", "mole"):
            if v == "own" and d not in OWN_DOMAINS:
                continue
            bjobs[(d, v)] = bench.spawn(d, v, max_new_tokens=bench_max_new_tokens,
                                        limit=bench_limit)
    out["bench"] = {f"{d}:{v}": c.get() for (d, v), c in bjobs.items()}
    print("[orchestrate] bench done", flush=True)

    agg = aggregate.remote()
    out["report"] = agg["_report.md"]
    print("\n" + agg["_report.md"], flush=True)
    return out


@app.local_entrypoint()
def run(epochs: int = 3, bench_limit: int = 100, skip_prep: bool = False):
    out = orchestrate.remote(epochs=epochs, bench_limit=bench_limit,
                             skip_prep=skip_prep)
    print(out.get("report", out))


@app.local_entrypoint()
def launch(epochs: int = 3, bench_limit: int = 100, skip_prep: bool = False,
           skip_train: bool = False):
    """Fire-and-forget spawn (use with --detach on a flaky network)."""
    call = orchestrate.spawn(epochs=epochs, bench_limit=bench_limit,
                             skip_prep=skip_prep, skip_train=skip_train)
    print(f"LAUNCHED orchestrate: {call.object_id}")


@app.local_entrypoint()
def agg_only():
    print(aggregate.remote()["_report.md"])


@app.local_entrypoint()
def smoke():
    """Cheap validation: tiny prep (8 prompts/domain), 1-epoch mole + comb_r16 +
    one own specialist, 2-prompt bench of lang_german across 4 variants, aggregate."""
    import json

    print("=== SMOKE prep (8 prompts/domain) ===")
    print(json.dumps(prep.remote(max_new_tokens=96, limit=8), indent=2))
    print("=== SMOKE train mole + comb_r16 + own lang_german (1 epoch) ===")
    jobs = {"mole": train_mole.spawn(epochs=1, val_every=1000),
            "comb_r16": train_lora.spawn("comb_r16", TRAIN_DOMAINS, epochs=1,
                                         val_every=1000),
            "lang_german": train_lora.spawn("lang_german", ["lang_german"],
                                            epochs=1, val_every=1000)}
    for name, c in jobs.items():
        print(f"[{name}]", c.get().get("saved"))
    print("=== SMOKE bench lang_german x base/own/comb_r16/mole (2 prompts) ===")
    bj = {v: bench.spawn("lang_german", v, max_new_tokens=64, limit=2, warmup=1)
          for v in ("base", "own", "comb_r16", "mole")}
    for v, c in bj.items():
        print(v, json.dumps(c.get(), indent=2))
    print("=== SMOKE aggregate ===")
    print(aggregate.remote()["_report.md"])
