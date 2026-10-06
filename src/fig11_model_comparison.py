import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
models    = ["LogReg", "KNN", "DT", "RF", "GB", "SVM", "NB", "MLP"]
f1_scores = [0.6645, 0.8118, 0.9993, 0.9993, 0.9993, 0.7681, 0.6618, 0.8574]
roc_auc   = [0.9948, 0.9048, 0.9989, 1.0000, 0.9988, 0.9929, 0.9869, 0.9691]

WIDTH = 0.35
OFFSET = WIDTH / 2
HIGHLIGHT = "SVM"

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(models))

bars_f1 = ax.bar(
    x - OFFSET,
    f1_scores,
    WIDTH,
    color="steelblue",
    label="F1-macro",
)

bars_auc = ax.bar(
    x + OFFSET,
    roc_auc,
    WIDTH,
    color="darkorange",
    label="ROC-AUC",
)

i_svm = models.index(HIGHLIGHT)
bars_f1[i_svm].set_edgecolor("red")
bars_f1[i_svm].set_linewidth(2.0)

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_xlabel("Classifier")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
ax.set_axisbelow(True)
ax.legend(loc="upper right")

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/fig11_model_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Figure 11 saved")
