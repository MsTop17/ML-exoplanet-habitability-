import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig, ax = plt.subplots(figsize=(8, 12))
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

BOX_X = 0.14
BOX_W = 0.72
BOX_H = 0.10
Y_TOP = [0.88, 0.75, 0.62, 0.49, 0.36, 0.23, 0.10]

steps = [
    "Raw Data\n5,832 confirmed exoplanets\nfrom NASA Exoplanet Archive",
    "Missing Value Handling\nDrop rows with >50% NaN\nKNNImputer (k=5)",
    "Log Transformation\nlog1p on pl_insol, pl_rade",
    "Feature Engineering\n+ greenhouse_index  + hz_score  + ESI",
    "Two-Stage Labeling\nStage 1: filter gas giants\nStage 2: 4 rocky climate bins",
    "Train/Test Split\n80/20 stratified\nStandardScaler (fit on train only)",
    "SMOTE\nApplied on training set only",
]

for y, text in zip(Y_TOP, steps):
    ax.add_patch(
        FancyBboxPatch(
            (BOX_X, y),
            BOX_W,
            BOX_H,
            boxstyle="round,pad=0.01",
            facecolor="lightblue",
            edgecolor="navy",
            alpha=0.3,
            linewidth=1.5,
        )
    )
    ax.text(
        BOX_X + BOX_W / 2,
        y + BOX_H / 2,
        text,
        ha="center",
        va="center",
        fontsize=10,
        linespacing=1.3,
    )

cx = BOX_X + BOX_W / 2
for y_top, y_next in zip(Y_TOP[:-1], Y_TOP[1:]):
    ax.add_patch(
        FancyArrowPatch(
            (cx, y_next + BOX_H),
            (cx, y_top),
            arrowstyle="-|>",
            mutation_scale=12,
            color="navy",
            linewidth=1.5,
        )
    )

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/fig7_pipeline.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Figure 7 saved")
