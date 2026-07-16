"""
Run the DFlash domain benchmark on Modal (GPU in the cloud).

Everything is persisted to two Modal Volumes so results are reusable later:
  - `dflash-bench-results`  -> /data   (all results JSONL/CSV/reports; durable)
  - `dflash-hf-cache`       -> /cache  (HuggingFace model weights; cached across runs)

The results volume is also synced back into this local `results/` folder by the
local entrypoint at the end of the run.

Usage (from the "Benchmarking domains" folder):

  export PATH="$HOME/Library/Python/3.11/bin:$PATH"

  # smoke test (2 domains x 5 prompts)
  modal run modal_run.py --limit 5 --max-new-tokens 256 --categories "lang_english code_python" --run-name smoke

  # full run (all 28 domains x 100 prompts)
  modal run modal_run.py --limit 100 --max-new-tokens 512 --categories all --run-name dflash_bench

  # inspect / reuse persisted data later
  modal volume ls   dflash-bench-results results
  modal volume get  dflash-bench-results results ./   # re-download everything
"""
import subprocess
from pathlib import Path

import modal

APP_NAME = "dflash-domain-benchmark"
GPU = "A100-40GB"          # 40GB: comfortably fits 8B target + 1B drafter + KV cache
TIMEOUT_S = 8 * 60 * 60    # 8h ceiling; a full run with baseline can be long

_HERE = Path(__file__).resolve().parent          # .../Benchmarking domains/scripts
_ROOT = _HERE.parent                              # .../Benchmarking domains
_DATA_ROOT = _ROOT / "data"                       # shared datasets: synthetic/ + downloaded/
_RESULTS_DIR = _ROOT / "results"                  # where local copies land
_CONTAINER_DATA = "/root/data"                    # shipped data root in the image

# Pinned to the versions the model card was evaluated with (+ datasets, needed at
# import; + matplotlib to render charts inside the run).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.9.1",
        "transformers==4.57.3",
        "accelerate>=1.0.0",
        "datasets>=3.0.0",
        "matplotlib>=3.7",
    )
    .env({"HF_HOME": "/cache", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    # ship the benchmark scripts into the image (paths are absolute, so this
    # works no matter which directory `modal run` is invoked from)
    .add_local_file(str(_HERE / "prompts.py"), "/root/prompts.py")
    .add_local_file(str(_HERE / "spec_patch.py"), "/root/spec_patch.py")
    .add_local_file(str(_HERE / "benchmark.py"), "/root/benchmark.py")
    .add_local_file(str(_HERE / "aggregate.py"), "/root/aggregate.py")
    .add_local_file(str(_HERE / "make_charts.py"), "/root/make_charts.py")
    # ship the shared datasets (data/synthetic + data/downloaded) so the benchmark can
    # read the per-domain test splits inside the container
    .add_local_dir(str(_DATA_ROOT), _CONTAINER_DATA)
)

results_vol = modal.Volume.from_name("dflash-bench-results", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("dflash-hf-cache", create_if_missing=True)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT_S,
    volumes={"/data": results_vol, "/cache": hf_cache_vol},
)
def run_benchmark(
    limit: int = 100,
    max_new_tokens: int = 512,
    categories: str = "all",
    run_name: str = "dflash_bench",
    split: str = "test",
    source: str = "synthetic",
    no_baseline: bool = False,
    resume: bool = True,
):
    """Runs benchmark.py + aggregate.py + make_charts.py inside the GPU container.

    Benchmarks the `split` (default: held-out test) of each domain in the chosen
    `source` selects the data/ subtree: synthetic (DataGen/Claude), wild (sorted
    WildChat), or downloaded (purpose-built HF datasets). Writes to the durable volume.
    """
    import torch

    print("GPU:", torch.cuda.get_device_name(0),
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB",
          "| torch", torch.__version__)

    out_dir = "/data/results"
    cmd = [
        "python", "/root/benchmark.py",
        "--run-name", run_name,
        "--limit", str(limit),
        "--max-new-tokens", str(max_new_tokens),
        "--out-dir", out_dir,
        "--prompt-source", "datagen",
        "--split", split,
        "--datagen-dir", f"{_CONTAINER_DATA}/{source}",
        "--categories", *categories.split(),
    ]
    if no_baseline:
        cmd.append("--no-baseline")
    if resume:
        cmd.append("--resume")

    print("[modal] running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd="/root", check=True)
    # persist raw results before aggregating, so a crash mid-aggregate loses nothing
    results_vol.commit()

    jsonl = f"{out_dir}/{run_name}.jsonl"
    subprocess.run(["python", "/root/aggregate.py", jsonl], cwd="/root", check=True)
    subprocess.run(["python", "/root/make_charts.py",
                    f"{out_dir}/{run_name}_by_category.csv"], cwd="/root", check=False)
    results_vol.commit()

    report_path = f"{out_dir}/{run_name}_report.md"
    csv_path = f"{out_dir}/{run_name}_by_category.csv"
    with open(report_path) as f:
        report = f.read()
    with open(jsonl) as f:
        jsonl_text = f.read()
    with open(csv_path) as f:
        csv_text = f.read()
    print("\n" + report)
    return {"report": report, "csv": csv_text, "jsonl": jsonl_text, "run_name": run_name}


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT_S,
    volumes={"/data": results_vol, "/cache": hf_cache_vol},
)
def run_shard(categories: list, limit: int, max_new_tokens: int, run_name: str,
              shard_id: int, split: str = "test", source: str = "synthetic"):
    """One container handles a subset of domains, writing its own shard file."""
    import torch
    shard_run = f"{run_name}_shard{shard_id}"
    print(f"[shard {shard_id}] {torch.cuda.get_device_name(0)} -> {categories}", flush=True)
    cmd = [
        "python", "/root/benchmark.py",
        "--run-name", shard_run,
        "--limit", str(limit),
        "--max-new-tokens", str(max_new_tokens),
        "--out-dir", "/data/results",
        "--prompt-source", "datagen",
        "--split", split,
        "--datagen-dir", f"{_CONTAINER_DATA}/{source}",
        "--resume",
        "--categories", *categories,
    ]
    subprocess.run(cmd, cwd="/root", check=True)
    results_vol.commit()
    return shard_run


@app.function(image=image, timeout=1800, volumes={"/data": results_vol})
def merge_and_aggregate(run_name: str):
    """Concatenate all shard JSONLs into one file and run aggregate.py over it."""
    import glob
    results_vol.reload()
    out_dir = "/data/results"
    shard_files = sorted(glob.glob(f"{out_dir}/{run_name}_shard*.jsonl"))
    merged = f"{out_dir}/{run_name}.jsonl"
    n = 0
    with open(merged, "w") as fout:
        for sf in shard_files:
            with open(sf) as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line)
                        n += 1
    print(f"[merge] {len(shard_files)} shards -> {merged} ({n} rows)")
    subprocess.run(["python", "/root/aggregate.py", merged], cwd="/root", check=True)
    subprocess.run(["python", "/root/make_charts.py",
                    f"{out_dir}/{run_name}_by_category.csv"], cwd="/root", check=False)
    results_vol.commit()
    with open(f"{out_dir}/{run_name}_report.md") as f:
        report = f.read()
    with open(f"{out_dir}/{run_name}_by_category.csv") as f:
        csv_text = f.read()
    with open(merged) as f:
        jsonl_text = f.read()
    print("\n" + report)
    return {"report": report, "csv": csv_text, "jsonl": jsonl_text, "run_name": run_name}


def _save_local(out):
    import os
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    rn = out["run_name"]
    for name, key in [(f"{rn}_report.md", "report"),
                      (f"{rn}_by_category.csv", "csv"),
                      (f"{rn}.jsonl", "jsonl")]:
        path = _RESULTS_DIR / name
        with open(path, "w") as f:
            f.write(out[key])
        print("[local] wrote", path)
    print("\nAll results (incl. charts/) persisted to Modal volume 'dflash-bench-results'.")
    print("Pull charts locally with:  modal volume get dflash-bench-results "
          "results ../results")


def _datagen_domains(source: str):
    """Domain names = subdirectories of data/<source> (must be generated first)."""
    src = _DATA_ROOT / source
    if not src.exists():
        raise SystemExit(
            f"No datasets at {src}.\n"
            f"Generate them first:  cd ../DataGen && python generate.py --group all")
    return sorted(p.name for p in src.iterdir() if p.is_dir())


@app.local_entrypoint()
def main(
    limit: int = 100,
    max_new_tokens: int = 512,
    categories: str = "all",
    run_name: str = "dflash_bench",
    split: str = "test",
    source: str = "synthetic",
    no_baseline: bool = False,
    resume: bool = True,
):
    """Single-container run (good for smoke tests / subsets)."""
    out = run_benchmark.remote(
        limit=limit, max_new_tokens=max_new_tokens, categories=categories,
        run_name=run_name, split=split, source=source,
        no_baseline=no_baseline, resume=resume,
    )
    _save_local(out)


@app.local_entrypoint()
def full(
    limit: int = 100,
    max_new_tokens: int = 512,
    run_name: str = "dflash_bench",
    split: str = "test",
    source: str = "synthetic",
    shards: int = 8,
):
    """Parallel run: shard all domains across `shards` GPU containers, then merge."""
    all_cats = _datagen_domains(source)
    groups = [all_cats[i::shards] for i in range(shards)]          # round-robin shard
    groups = [g for g in groups if g]
    print(f"[full] {len(all_cats)} domains x {limit} prompts across {len(groups)} shards "
          f"(source={source} split={split}):")
    for i, g in enumerate(groups):
        print(f"   shard {i}: {g}")

    tasks = [(g, limit, max_new_tokens, run_name, i, split, source)
             for i, g in enumerate(groups)]
    done = list(run_shard.starmap(tasks))                          # runs shards in parallel
    print(f"[full] shards complete: {done}")

    out = merge_and_aggregate.remote(run_name)
    _save_local(out)
