"""Instrumented *vanilla* (two-model) speculative decoding.

The original speculative-decoding recipe (Leviathan et al. 2023 / Chen et al.
2023): a completely separate small causal LM drafts k tokens autoregressively,
the frozen target verifies them in a single forward pass, and the longest
matching prefix (plus one correction token from the target) is committed. The
drafter never sees the target's hidden states — unlike EAGLE (conditions on
target features) or DFlash (conditions on target context features) — which is
exactly what experiment 06 probes: a drafter with no feature-alignment contract
should be the easiest one to specialize.

Greedy only (temperature 0): a proposal is accepted iff it equals the target's
argmax at that position, so the emitted tokens are byte-identical to the
target's own greedy decode — lossless by construction, same guarantee the
DFlash/EAGLE benches verify with `exact_match`.

Interface mirrors lib/spec_patch.py: `spec_generate(...)` returns
`(output_ids, acceptance_lengths)` where `acceptance_lengths[i]` is the number
of tokens committed on decode step i (accepted drafts + 1 correction/bonus
token, range 1..k+1). Proposed tokens per step is always exactly k.
"""
import torch
from transformers.cache_utils import DynamicCache


@torch.inference_mode()
def spec_generate(target, draft, input_ids, max_new_tokens, stop_token_ids,
                  k: int = 4, temperature: float = 0.0):
    """Two-model greedy speculative decode.

    target, draft: causal LMs sharing one tokenizer/vocab (asserted).
    input_ids: (1, n) prompt tokens on the models' device.
    Returns (output_ids, acceptance_lengths) — output truncated at the first
    stop token in the generated region and at max_new_tokens.
    """
    assert temperature == 0.0, "vanilla_spec is greedy-only (lossless bench)"
    assert target.config.vocab_size == draft.config.vocab_size, \
        "draft/target vocab mismatch — verification would be meaningless"
    target.eval(); draft.eval()

    n_in = input_ids.shape[1]
    max_length = n_in + max_new_tokens
    stop = set(int(s) for s in (stop_token_ids or []))

    t_cache, d_cache = DynamicCache(), DynamicCache()

    # Prefill: target processes the prompt and commits the first token.
    out = target(input_ids=input_ids, past_key_values=t_cache, use_cache=True,
                 logits_to_keep=1)
    seq = torch.cat([input_ids, out.logits[:, -1:].argmax(-1)], dim=1)
    # Invariants at the top of each decode step:
    #   t_cache covers seq[:, :-1]   (the last committed token is unverified)
    #   d_cache covers a prefix of seq[:, :-1] of length d_len
    d_len = 0
    acceptance_lengths = []

    while seq.shape[1] < max_length and int(seq[0, -1]) not in stop:
        # 1) draft proposes k tokens autoregressively (first forward consumes
        #    the uncached suffix of `seq`, each next consumes its own proposal)
        props = []
        inp = seq[:, d_len:]
        for _ in range(k):
            dout = draft(input_ids=inp, past_key_values=d_cache, use_cache=True,
                         logits_to_keep=1)
            inp = dout.logits[:, -1:].argmax(-1)
            props.append(inp)
        props = torch.cat(props, dim=1)                       # (1, k)

        # 2) target verifies [last committed, p1..pk] in one forward
        L = seq.shape[1]
        ver_in = torch.cat([seq[:, -1:], props], dim=1)       # (1, k+1)
        tout = target(input_ids=ver_in, past_key_values=t_cache, use_cache=True)
        greedy = tout.logits.argmax(-1)                       # greedy[:, i] follows ver_in[:, i]

        # 3) accept the longest matching prefix + 1 correction token
        j = int((props[0] == greedy[0, :-1]).cumprod(0).sum().item())
        committed = torch.cat([props[:, :j], greedy[:, j:j + 1]], dim=1)
        seq = torch.cat([seq, committed], dim=1)
        acceptance_lengths.append(j + 1)

        # 4) roll both caches back to the new committed prefix (minus the
        #    fresh correction token, which stays unverified)
        t_cache.crop(L + j)
        d_cache.crop(L + j)          # no-op when the cache is already shorter
        d_len = d_cache.get_seq_length()

        if any(int(t) in stop for t in committed[0].tolist()):
            break

    output_ids = seq[:, :max_length]
    if stop:
        gen = output_ids[0, n_in:].tolist()
        for i, t in enumerate(gen):
            if t in stop:
                output_ids = output_ids[:, :n_in + i + 1]
                break
    return output_ids, acceptance_lengths
