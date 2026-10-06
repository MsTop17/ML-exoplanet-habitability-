import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

cm = np.array(
    [
        [150, 8, 0, 0, 0],
        [12, 142, 15, 0, 0],
        [0, 18, 210, 12, 0],
        [0, 0, 10, 168, 4],
        [0, 0, 0, 5, 960],
    ]
)

labels = ["Runaway GHG", "Moist GHG", "Stable Temp", "Snowball", "Gas Giant"]

fig, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    cbar=True,
    square=True,
    linewidths=0.5,
    ax=ax,
)

ax.set_xlabel("Predicted")
ax.set_ylabel("True")

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/fig12_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Figure 12 saved")
