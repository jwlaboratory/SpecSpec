"""Serve N NaRA/LoRA adapters over one shared DFlash backbone, on a Modal GPU.

Separate Modal app from the benchmark (`dflash-domain-benchmark`) so this never
touches the running dflash_bench / diagnose jobs. Reuses the SAME image + the
`dflash-hf-cache` volume, so the 8B target + 1B drafter weights are already cached.

  # dry run with 5 toy adapters (proves the shared-backbone swap on real models)
  modal run modal_serve_nara.py --n-toy 5 --max-new-tokens 128

  # once the real adapter lands on the adapters volume:
  modal run modal_serve_nara.py --adapter-glob '/adapters/*.pt' --n-toy 4
"""
import glob as _glob
import os

import modal

APP_NAME = "nara-multi-adapter-serve"          # distinct from the benchmark app
GPU = "A100-40GB"                               # fits 8B target + 1B draft + KV
TIMEOUT_S = 60 * 60

# Identical pins to the benchmark image (versions the DFlash card was eval'd with).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.9.1",
        "transformers==4.57.3",
        "accelerate>=1.0.0",
        "datasets>=3.0.0",   # z-lab DFlash remote code imports `datasets` at load
    )
    .env({"HF_HOME": "/cache", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_file("../train/nara.py", "/root/nara.py")
    .add_local_file("adapter_bank.py", "/root/adapter_bank.py")
    .add_local_file("serve_nara.py", "/root/serve_nara.py")
)

hf_cache_vol = modal.Volume.from_name("dflash-hf-cache", create_if_missing=True)
# where the training agent's nara_adapter.pt is expected to be dropped (poll target)
adapters_vol = modal.Volume.from_name("nara-adapters", create_if_missing=True)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT_S,
    volumes={"/cache": hf_cache_vol, "/adapters": adapters_vol},
)
def serve(n_toy: int = 5, adapter_glob: str = "", max_new_tokens: int = 128, rank: int = 32):
    import torch
    from serve_nara import AdapterServer, load_backbone, make_demo_requests

    print("GPU:", torch.cuda.get_device_name(0),
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB | torch", torch.__version__)

    # ---- shared backbone: loaded ONCE, resident for every request/adapter ----
    draft, target, tok = load_backbone(attn="sdpa", device="cuda")
    server = AdapterServer(draft, target, tok, block_size=draft.config.block_size)

    # ---- adapters: real (from volume) + optional toy fillers ----
    real = sorted(_glob.glob(adapter_glob)) if adapter_glob else []
    if real:
        server.load_adapter_paths(real)   # any format: NaRA nara_state_dict or PEFT
        print(f"[modal] loaded {len(real)} REAL adapters: {[os.path.basename(p) for p in real]}")
    if n_toy:
        server.add_toy_adapters(n_toy, rank=rank)
    if server.bank.num_adapters == 0:
        raise SystemExit("No adapters. Pass --n-toy N or --adapter-glob.")

    # ---- 5 requests, each routed to a different adapter, over the one backbone ----
    reqs = make_demo_requests(server.bank.num_adapters, max_new_tokens=max_new_tokens)
    results = server.serve_batch(reqs)

    print("\n" + "=" * 72)
    print(f"SHARED BACKBONE: 1x Qwen3-8B target + 1x DFlash drafter (loaded once)")
    print(f"ADAPTERS RESIDENT: {server.bank.num_adapters}   REQUESTS SERVED: {len(results)}")
    for i, r in enumerate(results):
        print(f"\n[req {i}] adapter='{r.adapter_name}'  {r.n_tokens} tok  "
              f"{r.tokens_per_sec:.1f} tok/s")
        print("   " + r.text.replace("\n", "\n   ")[:400])
    print("=" * 72)
    return {
        "num_adapters": server.bank.num_adapters,
        "results": [(r.adapter_name, r.n_tokens, r.tokens_per_sec) for r in results],
    }


@app.local_entrypoint()
def main(n_toy: int = 5, adapter_glob: str = "", max_new_tokens: int = 128, rank: int = 32):
    out = serve.remote(n_toy=n_toy, adapter_glob=adapter_glob,
                       max_new_tokens=max_new_tokens, rank=rank)
    print("\n[local] served:", out["num_adapters"], "adapters")
    for name, ntok, tps in out["results"]:
        print(f"  {name:16s} {ntok:4d} tok  {tps:6.1f} tok/s")
