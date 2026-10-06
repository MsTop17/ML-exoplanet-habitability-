import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

features = [
    "pl_insol",
    "pl_rade",
    "pl_eqt",
    "st_teff",
    "st_mass",
    "co2_frac",
    "h2o_frac",
    "ch4_frac",
    "greenhouse_index",
    "esi",
]
deltas = [0.32, 0.18, 0.05, 0.03, 0.02, 0.04, 0.06, 0.03, 0.09, 0.07]

HIGHLIGHT = "pl_insol"

order = np.argsort(deltas)
features = [features[i] for i in order]
deltas = [deltas[i] for i in order]

colors = ["red" if f == HIGHLIGHT else "steelblue" for f in features]

fig, ax = plt.subplots(figsize=(9, 6))

y = np.arange(len(features))
ax.barh(y, deltas, color=colors, edgecolor="black", linewidth=0.6)

ax.set_yticks(y)
ax.set_yticklabels(features)
ax.set_xlabel("Drop in F1-macro when feature is removed")
ax.set_ylabel("Feature")
ax.grid(True, axis="x", alpha=0.25, linewidth=0.5)
ax.set_axisbelow(True)

for yi, d in zip(y, deltas):
    ax.text(
        d + max(deltas) * 0.015,
        yi,
        f"{d:.2f}",
        va="center",
        fontsize=9,
    )

ax.set_xlim(0, max(deltas) * 1.12)

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/fig14_ablation.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Figure 14 saved")
