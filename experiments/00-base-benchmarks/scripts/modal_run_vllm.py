"""
Run the unified vLLM speculator benchmark on Modal (GPU in the cloud).

One container per (method, source) — vLLM batches all of a source's domains
internally, so no sharding needed. Both speculators run the SAME prompts:

  modal run modal_run_vllm.py::main --method eagle3 --source synthetic
  modal run modal_run_vllm.py::main --method dflash --source wild
  modal run modal_run_vllm.py::all           # both methods x all 3 sources

Results (per-domain acceptance rate + mean accept length + charts) sync back to
../results and persist on the `dflash-bench-results` volume.

vLLM nightly is required for DFlash (`method: "dflash"`); EAGLE3 works on it too.
"""
import subprocess
from pathlib import Path

import modal

APP_NAME = "spec-domain-benchmark-vllm"
GPU = "A100-40GB"          # 8B target + 1B speculator + KV cache, bf16
TIMEOUT_S = 8 * 60 * 60

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_DATA_ROOT = _ROOT.parent.parent / "data"
_RESULTS_DIR = _ROOT / "results"
_CONTAINER_DATA = "/root/data"

# CUDA *devel* base so flashinfer (vLLM's sampler) has nvcc for its JIT compile —
# debian_slim lacks nvcc and vLLM fails engine init with "Could not find nvcc".
# We also force FLASH_ATTN attention + disable the flashinfer sampler to minimise
# runtime JIT. vLLM nightly is needed for DFlash's `method: "dflash"`.
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .run_commands(
        "pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly",
        "pip install -U transformers matplotlib hf_transfer",
    )
    .env({
        "HF_HOME": "/cache",
        "CUDA_HOME": "/usr/local/cuda",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    })
    .add_local_file(str(_HERE / "benchmark_vllm.py"), "/root/benchmark_vllm.py")
    .add_local_file(str(_HERE / "make_charts.py"), "/root/make_charts.py")
    .add_local_dir(str(_DATA_ROOT), _CONTAINER_DATA)
)

results_vol = modal.Volume.from_name("dflash-bench-results", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("dflash-hf-cache", create_if_missing=True)

app = modal.App(APP_NAME)


@app.function(image=image, gpu=GPU, timeout=TIMEOUT_S,
              volumes={"/data": results_vol, "/cache": hf_cache_vol})
def run_bench(method: str, source: str, split: str = "test",
              limit: int = None, max_new_tokens: int = 512):
    """Run benchmark_vllm.py for one (method, source) → CSV + report + charts."""
    import torch
    print("GPU:", torch.cuda.get_device_name(0), "| method", method, "| source", source)

    run_name = f"{method}_{source}"
    out_dir = "/data/results"
    cmd = [
        "python", "/root/benchmark_vllm.py",
        "--method", method,
        "--datagen-dir", f"{_CONTAINER_DATA}/{source}",
        "--split", split,
        "--run-name", run_name,
        "--out-dir", out_dir,
        "--max-new-tokens", str(max_new_tokens),
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    print("[modal] running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd="/root", check=True)

    subprocess.run(["python", "/root/make_charts.py",
                    f"{out_dir}/{run_name}_by_category.csv"], cwd="/root", check=False)
    results_vol.commit()

    csv_text = Path(f"{out_dir}/{run_name}_by_category.csv").read_text()
    report = Path(f"{out_dir}/{run_name}_report.md").read_text()
    print("\n" + report)
    return {"run_name": run_name, "csv": csv_text, "report": report}


def _save_local(out):
    import os
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    rn = out["run_name"]
    (_RESULTS_DIR / f"{rn}_by_category.csv").write_text(out["csv"])
    (_RESULTS_DIR / f"{rn}_report.md").write_text(out["report"])
    print(f"[local] wrote {rn}_by_category.csv + _report.md to {_RESULTS_DIR}")
    print("Charts persisted on volume 'dflash-bench-results'. Pull with:\n"
          "  modal volume get dflash-bench-results results ../results")


@app.local_entrypoint()
def main(method: str = "eagle3", source: str = "synthetic",
         split: str = "test", limit: int = None, max_new_tokens: int = 512):
    """One (method, source) run."""
    out = run_bench.remote(method=method, source=source, split=split,
                           limit=limit, max_new_tokens=max_new_tokens)
    _save_local(out)


@app.local_entrypoint()
def all(split: str = "test", limit: int = None, max_new_tokens: int = 512):
    """Both speculators × all three sources, in parallel."""
    methods = ["dflash", "eagle3"]
    sources = ["synthetic", "wild", "downloaded"]
    tasks = [(m, s, split, limit, max_new_tokens) for m in methods for s in sources]
    print(f"[all] {len(tasks)} runs: {methods} × {sources}")
    for out in run_bench.starmap(tasks):
        _save_local(out)


if __name__ == "__main__":
    main()
