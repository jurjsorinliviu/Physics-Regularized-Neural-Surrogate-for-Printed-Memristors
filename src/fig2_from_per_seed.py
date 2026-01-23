import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

root = Path(r"C:\Users\jurjs\Desktop\PostDoc pmTUC\Papers ideas\PRNS for printed memristors paper")
csv_path = root / "results/cv_8seeds/per_seed_metrics.csv"
out_path = root / "IEEE Access Version/figures/cross_validation_rrmse.png"

prns = []
vteam = []
with csv_path.open() as f:
    r = csv.DictReader(f)
    for row in r:
        prns.append(float(row["pinn_rrmse"]))
        vteam.append(float(row["vteam_rrmse"]))

fig, ax = plt.subplots(figsize=(6, 4))

# PRNS boxplot only
bp = ax.boxplot([prns], positions=[1], widths=0.5, showfliers=True,
                boxprops=dict(linewidth=1.5),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
                medianprops=dict(linewidth=2))

# scatter PRNS points
x1 = np.random.normal(1, 0.04, len(prns))
ax.scatter(x1, prns, s=30, alpha=0.7, color="orange")

# VTEAM as centered horizontal line (no variance)
vteam_val = vteam[0]
ax.hlines(vteam_val, 1.85, 2.15, colors="#1f77b4", linewidth=2)

ax.set_xticks([1, 2])
ax.set_xticklabels(["PRNS", "VTEAM"])
ax.set_xlabel("Model")
ax.set_ylabel("RRMSE")
ax.set_title("Cross-Validation RRMSE Summary (8 seeds)")
ax.set_xlim(0.5, 2.5)
ax.set_ylim(0.0, 0.30)

# legend
legend_handles = [
    Patch(facecolor="white", edgecolor="black", label="PRNS (boxplot)"),
    Line2D([0], [0], color="#1f77b4", lw=2, label="VTEAM (constant)")
]
ax.legend(handles=legend_handles, loc="upper left", frameon=True)

fig.tight_layout()
fig.savefig(out_path, dpi=600, bbox_inches="tight")

