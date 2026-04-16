import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/mpl")))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache")))

import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent

def resolve_latest_run(folder: Path) -> Path:
    latest_run_file = folder / "LATEST_RUN"
    if latest_run_file.exists():
        run_path = Path(latest_run_file.read_text().strip())
        if run_path.exists():
            return run_path
        return folder / run_path.name
    subdirs = [p for p in folder.iterdir() if p.is_dir()]
    return sorted(subdirs)[-1]

baseline_dir = resolve_latest_run(BASE / "baseline_results")
naive_dir = resolve_latest_run(BASE / "serverless_naive_results")
keepwarm_dir = resolve_latest_run(BASE / "serverless_keepwarm_results")

plot_targets = {
    "baseline": baseline_dir / "gpu_monitor.csv",
    "naive_serverless": naive_dir / "gpu_monitor.csv",
    "keep_warm_serverless": keepwarm_dir / "gpu_monitor.csv",
}

Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

output_dir = BASE / "plots"
output_dir.mkdir(exist_ok=True)

for label, csv_path in plot_targets.items():
    df = pd.read_csv(csv_path)

    df.columns = [c.strip() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["memory.used [MiB]"] = (
        df["memory.used [MiB]"].astype(str).str.replace("MiB", "", regex=False).str.strip().astype(float)
    )
    df["elapsed_sec"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()

    plt.figure()
    plt.plot(df["elapsed_sec"], df["memory.used [MiB]"])
    plt.xlabel("Elapsed Time (s)")
    plt.ylabel("GPU Memory (MiB)")
    plt.title(f"{label}: GPU Memory Usage Over Time")
    plt.tight_layout()

    out_file = output_dir / f"{label}_gpu_memory.png"
    plt.savefig(out_file, dpi=200)
    plt.close()

    print(f"saved: {out_file}")
