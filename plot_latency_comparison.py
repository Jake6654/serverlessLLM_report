import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("/Users/jake/Documents/serverlessLLM_report")
output_dir = BASE / "figures"
output_dir.mkdir(exist_ok=True)

configs = ["Baseline\nWarm", "Naive\nServerless", "Keep-Warm\nServerless"]
latency = [0.0199, 0.540, 0.411]

plt.figure()
plt.bar(configs, latency)
plt.ylabel("Latency (seconds)")
plt.title("Latency Comparison Across Deployment Strategies")
plt.tight_layout()
plt.savefig(output_dir / "latency_comparison.png", dpi=200)
plt.close()

print("saved:", output_dir / "latency_comparison.png")