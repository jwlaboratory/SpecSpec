"""
Diagnostic: is the DFlash lossless-mismatch a bug in our harness, or expected
floating-point drift between parallel block-verification and sequential greedy?

For a few prompts, compute the continuation THREE ways using only the target:
  1. spec     -> DFlash spec_generate (block-parallel target verification)
  2. hf       -> target.generate(do_sample=False)  (HF sequential greedy)
  3. manual   -> our own one-token-at-a-time argmax loop (sequential greedy)

Expectation if there is NO bug:
  - hf == manual, bit-for-bit (both are sequential greedy; same fp path).
  - spec agrees with them for a long prefix, then diverges at a near-tie logit,
    where target's top-2 tokens are almost equal in logit value.

  modal run modal_diagnose.py
"""
import modal

# Self-contained image/volumes (mirrors modal_run.py so this file runs standalone).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.9.1", "transformers==4.57.3", "accelerate>=1.0.0", "datasets>=3.0.0")
    .env({"HF_HOME": "/cache", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_file("prompts.py", "/root/prompts.py")
    .add_local_file("spec_patch.py", "/root/spec_patch.py")
)
results_vol = modal.Volume.from_name("dflash-bench-results", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("dflash-hf-cache", create_if_missing=True)

app = modal.App("dflash-diagnose")


@app.function(image=image, gpu="A100-40GB", timeout=1800,
              volumes={"/data": results_vol, "/cache": hf_cache_vol})
def diagnose(max_new_tokens: int = 256):
    import sys
    sys.path.insert(0, "/root")
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    from spec_patch import make_instrumented_spec_generate
    from prompts import build_prompts

    dev = "cuda:0"
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    draft = AutoModel.from_pretrained("z-lab/Qwen3-8B-DFlash-b16", trust_remote_code=True,
                                      dtype="auto", attn_implementation="sdpa").to(dev).eval()
    target = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype="auto",
                                                  attn_implementation="sdpa").to(dev).eval()
    draft.spec_generate = make_instrumented_spec_generate(draft)
    eos = tok.eos_token_id

    def build(prompt):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        return tok([text], return_tensors="pt").input_ids.to(dev)

    @torch.inference_mode()
    def manual_greedy(input_ids, n):
        from transformers import DynamicCache
        cache = DynamicCache()
        out = target(input_ids, use_cache=True, past_key_values=cache)
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        gen = [nxt.item()]
        for _ in range(n - 1):
            if gen[-1] == eos:
                break
            out = target(nxt, use_cache=True, past_key_values=cache)
            nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
            gen.append(nxt.item())
        return gen

    def trim(ids):
        for i, t in enumerate(ids):
            if t == eos:
                return ids[:i + 1]
        return ids

    def first_div(a, b):
        m = min(len(a), len(b))
        for i in range(m):
            if a[i] != b[i]:
                return i
        return m if len(a) == len(b) else m

    prompts = build_prompts(100)
    picks = [("lang_english", 0), ("code_python", 2), ("lang_english", 2)]
    for cat, idx in picks:
        p = prompts[cat][idx]
        ids = build(p)
        n_in = ids.shape[1]

        spec_ids, _ = draft.spec_generate(target=target, input_ids=ids,
                                          max_new_tokens=max_new_tokens,
                                          stop_token_ids=[eos], temperature=0.0)
        spec = trim(spec_ids[0, n_in:].tolist())
        hf = trim(target.generate(input_ids=ids, max_new_tokens=max_new_tokens,
                                  do_sample=False, num_beams=1,
                                  pad_token_id=eos, eos_token_id=[eos],
                                  use_cache=True)[0, n_in:].tolist())
        man = trim(manual_greedy(ids, max_new_tokens))

        d_sh = first_div(spec, hf)
        d_sm = first_div(spec, man)
        d_hm = first_div(hf, man)
        print("\n" + "=" * 78)
        print(f"[{cat}#{idx}]  {p[:70]!r}")
        print(f"  lengths: spec={len(spec)} hf={len(hf)} manual={len(man)}")
        print(f"  first divergence:  spec-vs-hf={d_sh}   spec-vs-manual={d_sm}   hf-vs-manual={d_hm}")
        print(f"  --> hf == manual (both sequential greedy)? "
              f"{'YES (bit-identical)' if d_hm == min(len(hf), len(man)) and len(hf) == len(man) else 'NO'}")

        # Inspect the near-tie at the spec-vs-hf divergence, teacher-forcing the hf prefix.
        if d_sh < len(hf) and d_sh < len(spec):
            prefix = torch.tensor([ids[0].tolist() + hf[:d_sh]], device=dev)
            with torch.inference_mode():
                logits = target(prefix, use_cache=False).logits[0, -1, :].float()
            top = torch.topk(logits, 2)
            t0, t1 = top.indices.tolist()
            g0, g1 = top.values.tolist()
            print(f"  at divergence idx {d_sh}: spec chose {spec[d_sh]} "
                  f"({tok.decode([spec[d_sh]])!r}), hf chose {hf[d_sh]} ({tok.decode([hf[d_sh]])!r})")
            print(f"  target top-2 logits here: {t0}({tok.decode([t0])!r})={g0:.4f}  "
                  f"vs {t1}({tok.decode([t1])!r})={g1:.4f}   gap={abs(g0-g1):.5f}")
            print(f"  --> {'NEAR-TIE (fp drift, benign)' if abs(g0-g1) < 0.05 else 'LARGE GAP (investigate!)'}")

    return "diagnostic complete"


@app.local_entrypoint()
def main(max_new_tokens: int = 256):
    print(diagnose.remote(max_new_tokens=max_new_tokens))
