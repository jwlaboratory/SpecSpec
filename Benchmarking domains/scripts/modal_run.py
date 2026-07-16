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

import modal

APP_NAME = "dflash-domain-benchmark"
GPU = "A100-40GB"          # 40GB: comfortably fits 8B target + 1B drafter + KV cache
TIMEOUT_S = 8 * 60 * 60    # 8h ceiling; a full 100x28 run with baseline can be long

# Pinned to the versions the model card was evaluated with (+ datasets, needed at import).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.9.1",
        "transformers==4.57.3",
        "accelerate>=1.0.0",
        "datasets>=3.0.0",
    )
    .env({"HF_HOME": "/cache", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    # ship the benchmark scripts into the image
    .add_local_file("prompts.py", "/root/prompts.py")
    .add_local_file("spec_patch.py", "/root/spec_patch.py")
    .add_local_file("benchmark.py", "/root/benchmark.py")
    .add_local_file("aggregate.py", "/root/aggregate.py")
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
    no_baseline: bool = False,
    resume: bool = True,
):
    """Runs benchmark.py + aggregate.py inside the GPU container, writing to /data."""
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
def run_shard(categories: list, limit: int, max_new_tokens: int, run_name: str, shard_id: int):
    """One container handles a subset of categories, writing its own shard file."""
    import torch
    shard_run = f"{run_name}_shard{shard_id}"
    print(f"[shard {shard_id}] {torch.cuda.get_device_name(0)} -> {categories}", flush=True)
    cmd = [
        "python", "/root/benchmark.py",
        "--run-name", shard_run,
        "--limit", str(limit),
        "--max-new-tokens", str(max_new_tokens),
        "--out-dir", "/data/results",
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
    os.makedirs("results", exist_ok=True)
    rn = out["run_name"]
    for name, key in [(f"results/{rn}_report.md", "report"),
                      (f"results/{rn}_by_category.csv", "csv"),
                      (f"results/{rn}.jsonl", "jsonl")]:
        with open(name, "w") as f:
            f.write(out[key])
        print("[local] wrote", name)
    print("\nAll results also persisted to Modal volume 'dflash-bench-results' (/data/results).")
    print("Re-download later with:  modal volume get dflash-bench-results results ./")


@app.local_entrypoint()
def main(
    limit: int = 100,
    max_new_tokens: int = 512,
    categories: str = "all",
    run_name: str = "dflash_bench",
    no_baseline: bool = False,
    resume: bool = True,
):
    """Single-container run (good for smoke tests / subsets)."""
    out = run_benchmark.remote(
        limit=limit, max_new_tokens=max_new_tokens, categories=categories,
        run_name=run_name, no_baseline=no_baseline, resume=resume,
    )
    _save_local(out)


@app.local_entrypoint()
def full(
    limit: int = 100,
    max_new_tokens: int = 512,
    run_name: str = "dflash_bench",
    shards: int = 7,
):
    """Parallel run: shard all 28 domains across `shards` GPU containers, then merge."""
    from prompts import build_prompts
    all_cats = list(build_prompts(1).keys())
    groups = [all_cats[i::shards] for i in range(shards)]          # round-robin shard
    groups = [g for g in groups if g]
    print(f"[full] {len(all_cats)} domains x {limit} prompts across {len(groups)} shards:")
    for i, g in enumerate(groups):
        print(f"   shard {i}: {g}")

    tasks = [(g, limit, max_new_tokens, run_name, i) for i, g in enumerate(groups)]
    done = list(run_shard.starmap(tasks))                          # runs shards in parallel
    print(f"[full] shards complete: {done}")

    out = merge_and_aggregate.remote(run_name)
    _save_local(out)
