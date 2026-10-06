import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(11, 6))

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.01, 100)
ax.set_ylim(0.1, 20)

regimes = [
    (0.01, 0.35, "blue", "Snowball"),
    (0.35, 1.1, "green", "Stable Temperate"),
    (1.1, 1.5, "orange", "Moist Greenhouse"),
    (1.5, 100.0, "red", "Runaway Greenhouse"),
]

for s_lo, s_hi, color, label in regimes:
    ax.axvspan(s_lo, s_hi, color=color, alpha=0.25)
    ax.text(
        np.sqrt(s_lo * s_hi),
        0.13,
        label,
        ha="center",
        va="center",
        fontsize=10,
        color="black",
    )

ax.axhspan(1.6, 20, color="gray", alpha=0.20)
ax.text(
    np.sqrt(0.01 * 100),
    np.sqrt(1.6 * 20),
    "Gas Giant",
    ha="center",
    va="center",
    fontsize=11,
    color="black",
)

for s_edge in (0.35, 1.1, 1.5):
    ax.axvline(s_edge, linestyle="--", color="black", linewidth=1.2, alpha=0.7)

ax.axhline(1.6, linestyle="--", color="black", linewidth=1.2)
ax.text(
    45.0,
    1.85,
    "Fulton gap",
    ha="right",
    va="bottom",
    fontsize=10,
    color="black",
)

stars = [
    (1.0, 1.0, "Earth", (10, 10)),
    (0.43, 0.53, "Mars", (-10, 12)),
    (1.9, 0.95, "Venus", (12, -4)),
]

for s_val, r_val, label, offset in stars:
    ax.plot(
        s_val,
        r_val,
        marker="*",
        markersize=16,
        color="black",
        markeredgecolor="white",
        markeredgewidth=0.8,
        linestyle="none",
    )
    ax.annotate(
        label,
        xy=(s_val, r_val),
        xytext=offset,
        textcoords="offset points",
        fontsize=11,
        color="black",
    )

ax.set_xlabel("Stellar Insolation Flux S (S_earth)")
ax.set_ylabel("Planetary Radius R (R_earth)")
ax.grid(True, which="both", alpha=0.25, linewidth=0.5)

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/fig2_climate_regimes.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Figure 2 saved")