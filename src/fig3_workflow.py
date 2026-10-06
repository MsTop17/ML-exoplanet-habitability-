import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig, ax = plt.subplots(figsize=(12, 8))
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

BOX_W = 0.35
BOX_H = 0.13
Y_TOP = [0.82, 0.62, 0.42, 0.22]
X_LEFT = 0.05
X_RIGHT = 0.55

left_boxes = [
    "Data Collection\nNASA Exoplanet Archive\n5,832 planets",
    "Preprocessing\nKNN Imputer + Log Transform\n+ StandardScaler",
    "Feature Engineering\n10 primary + 2 derived features",
    "Label Creation\n5 climate stability bins",
]

right_boxes = [
    "Stage 1: Filter Gas Giants\n(R >= 1.6 R_earth)\nRemoves 4,714 planets",
    "Stage 2: Classify\n1,118 rocky candidates",
    "Train 8 Classifiers\n5-fold CV + GridSearchCV",
    "Evaluation\nF1-macro, ROC-AUC, SHAP",
]


def draw_box(x, y, text, facecolor, edgecolor):
    box = FancyBboxPatch(
        (x, y),
        BOX_W,
        BOX_H,
        boxstyle="round,pad=0.01",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.5,
    )
    ax.add_patch(box)
    ax.text(
        x + BOX_W / 2,
        y + BOX_H / 2,
        text,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        linespacing=1.3,
    )


def draw_arrow(start, end, **kwargs):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=1.5,
            **kwargs,
        )
    )


for y, text in zip(Y_TOP, left_boxes):
    draw_box(X_LEFT, y, text, "lightblue", "navy")

for y, text in zip(Y_TOP, right_boxes):
    draw_box(X_RIGHT, y, text, "lightgreen", "darkgreen")

for x in (X_LEFT, X_RIGHT):
    for y_top, y_next in zip(Y_TOP[:-1], Y_TOP[1:]):
        draw_arrow(
            (x + BOX_W / 2, y_next + BOX_H),
            (x + BOX_W / 2, y_top),
            color="black",
        )

draw_arrow(
    (X_LEFT + BOX_W, Y_TOP[-1] + BOX_H / 2),
    (X_RIGHT, Y_TOP[0] + BOX_H / 2),
    color="black",
    connectionstyle="arc3,rad=-0.2",
)

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/fig3_workflow.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Figure 3 saved")
