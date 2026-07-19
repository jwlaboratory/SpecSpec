"""
Instrumented copy of DFlash's `spec_generate` (vendored into finetuning/).

The upstream `spec_generate` (in the z-lab model repo's dflash.py) computes per-step
`acceptance_lengths` internally but throws them away, returning only `output_ids`.
We need those numbers to compute acceptance rate / mean accepted length, so this
module rebinds a byte-for-byte copy of the method that ALSO returns the
acceptance list.

The copy is kept intentionally identical to the original decode loop so the
generated tokens are exactly the same as upstream -- we only add bookkeeping.
Helper functions (`sample`, `extract_context_feature`, `DynamicCache`) are pulled
straight out of the model's own dynamically-loaded module so we never diverge
from the version that shipped with the weights.

Source reference (z-lab/Qwen3-8B-DFlash-b16 @ dflash.py, DFlashDraftModel.spec_generate).
"""
import sys
import types

import torch


def make_instrumented_spec_generate(model):
    """Return a bound method `spec_generate(target, input_ids, max_new_tokens,
    stop_token_ids, temperature)` that returns `(output_ids, acceptance_lengths)`.

    `acceptance_lengths[i]` is the number of tokens *committed* on decode step i,
    i.e. (accepted draft tokens + 1 bonus token). It ranges from 1..block_size.
    """
    mod = sys.modules[type(model).__module__]
    DynamicCache = mod.DynamicCache
    sample = mod.sample
    extract_context_feature = mod.extract_context_feature

    @torch.inference_mode()
    def spec_generate(self, target, input_ids, max_new_tokens, stop_token_ids, temperature):
        self.eval()
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens

        block_size = self.block_size
        output_ids = torch.full(
            (1, max_length + block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=target.device,
        )
        position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)

        past_key_values_target = DynamicCache()
        past_key_values_draft = DynamicCache()

        # Prefill stage
        output = target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )

        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens:num_input_tokens + 1] = sample(output.logits, temperature)
        target_hidden = extract_context_feature(output.hidden_states, self.target_layer_ids)

        # Decode stage
        acceptance_lengths = []
        start = input_ids.shape[1]
        while start < max_length:
            block_output_ids = output_ids[:, start:start + block_size].clone()
            block_position_ids = position_ids[:, start:start + block_size]
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_logits = target.lm_head(self(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, past_key_values_draft.get_seq_length(): start + block_size],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )[:, -block_size + 1:, :])
            past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = sample(draft_logits)

            output = target(
                block_output_ids,
                position_ids=block_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
            )

            posterior = sample(output.logits, temperature)
            acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
            output_ids[:, start:start + acceptance_length + 1] = block_output_ids[:, :acceptance_length + 1]
            output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
            start += acceptance_length + 1
            past_key_values_target.crop(start)
            target_hidden = extract_context_feature(output.hidden_states, self.target_layer_ids)[:, :acceptance_length + 1, :]
            acceptance_lengths.append(acceptance_length + 1)
            if stop_token_ids is not None and any(
                stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
            ):
                break
        output_ids = output_ids[:, :max_length]
        output_ids = output_ids[:, output_ids[0] != self.mask_token_id]
        if stop_token_ids is not None:
            stop_token_ids_t = torch.tensor(stop_token_ids, device=output_ids.device)
            stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids_t).nonzero(as_tuple=True)[0]
            if stop_token_indices.numel() > 0:
                output_ids = output_ids[:, :num_input_tokens + stop_token_indices[0] + 1]

        return output_ids, acceptance_lengths

    return types.MethodType(spec_generate, model)
