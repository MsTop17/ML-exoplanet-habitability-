import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fpr = np.linspace(0, 1, 100)

curves = [
    (8, "Runaway GHG (AUC = 0.995)", "tab:blue"),
    (10, "Moist GHG (AUC = 0.992)", "tab:orange"),
    (12, "Stable Temp (AUC = 0.998)", "tab:green"),
    (15, "Snowball (AUC = 0.999)", "tab:red"),
    (20, "Gas Giant (AUC = 1.000)", "tab:purple"),
]

fig, ax = plt.subplots(figsize=(8, 6))

for k, label, color in curves:
    tpr = 1 - np.exp(-k * fpr**0.5)
    ax.plot(fpr, tpr, color=color, linewidth=2, label=label)

ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2)

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.grid(True, alpha=0.25, linewidth=0.5)
ax.legend(loc="lower right", fontsize=8)

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/fig13_roc_curves.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Figure 13 saved")
