import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def s_eff(T_eff, coeffs):
    c0, c1, c2, c3, c4 = coeffs
    T = T_eff - 5780
    return c0 + c1 * T + c2 * T**2 + c3 * T**3 + c4 * T**4


recent_venus = (1.776, 2.136e-4, 2.533e-8, -1.332e-11, -3.097e-15)
runaway_ghg = (1.107, 1.332e-4, 1.580e-8, -8.308e-12, -1.931e-15)
max_ghg = (0.356, 6.171e-5, 1.698e-9, -3.198e-12, -5.575e-16)
early_mars = (0.320, 5.547e-5, 1.526e-9, -2.874e-12, -5.011e-16)

fig, ax = plt.subplots(figsize=(11, 6))

T = np.linspace(2600, 7200, 800)

S_venus = s_eff(T, recent_venus)
S_runaway = s_eff(T, runaway_ghg)
S_max = s_eff(T, max_ghg)
S_early = s_eff(T, early_mars)

ax.fill_between(
    T, S_runaway, S_max, color="lightgreen", alpha=0.3, label="Conservative HZ"
)
ax.fill_between(
    T, S_venus, S_early, color="lightyellow", alpha=0.15, label="Optimistic HZ"
)

ax.plot(T, S_venus, color="orange", linewidth=2, label="Recent Venus")
ax.plot(T, S_runaway, color="red", linewidth=2, label="Runaway Greenhouse")
ax.plot(T, S_max, color="blue", linewidth=2, label="Maximum Greenhouse")
ax.plot(T, S_early, color="cyan", linewidth=2, label="Early Mars")

for s_val, name, offset in [
    (1.0, "Earth", (-12, 12)),
    (0.43, "Mars", (-12, -20)),
    (1.9, "Venus", (-12, 14)),
]:
    ax.plot(
        5780,
        s_val,
        marker="*",
        markersize=16,
        color="black",
        markeredgecolor="white",
        markeredgewidth=0.8,
        linestyle="none",
    )
    ax.annotate(
        name,
        xy=(5780, s_val),
        xytext=offset,
        textcoords="offset points",
        fontsize=11,
        color="black",
    )

ax.set_xlim(2600, 7200)
ax.set_yscale("log")
ax.set_ylim(0.1, 100)

ax.set_xlabel("Stellar Effective Temperature T_eff (K)")
ax.set_ylabel("Stellar Insolation Flux S_eff (S_earth)")
ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
ax.legend(loc="upper right", fontsize=8)

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/fig4_hz_boundaries.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Figure 4 saved")
