import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

years = [1995, 1998, 2000, 2005, 2010, 2014, 2016, 2018, 2020, 2022, 2024]
counts = [1, 10, 50, 200, 500, 1800, 3400, 3800, 4300, 5000, 5500]

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    years,
    counts,
    color="steelblue",
    linewidth=2,
    markersize=6,
    marker="o",
)

for year, label in [(2009, "Kepler (2009)"), (2018, "TESS (2018)"), (2021, "JWST (2021)")]:
    ax.axvline(year, linestyle="--", color="gray")
    ax.text(
        year,
        ax.get_ylim()[1] * 0.95,
        label,
        rotation=90,
        va="top",
        ha="right",
        color="gray",
        fontsize=9,
    )

ax.plot(
    [years[-1]],
    [counts[-1]],
    marker="o",
    markersize=11,
    markerfacecolor="crimson",
    markeredgecolor="black",
    markeredgewidth=1.5,
    linestyle="none",
    zorder=5,
)

ax.annotate(
    "5,500+ (2024)",
    xy=(2024, 5500),
    xytext=(2016, 3900),
    arrowprops=dict(arrowstyle="->", color="black"),
    fontsize=10,
)

ax.set_xlim(1994, 2024.5)
ax.set_xlabel("Year")
ax.set_ylabel("Cumulative Confirmed Exoplanets")
ax.grid(True, alpha=0.3)

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/fig1_discovery_timeline.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Figure 1 updated")
