import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(13, 7))

bins = [
    (4, 1.5, 10.0, "red"),
    (3, 1.1, 1.5, "orange"),
    (2, 0.35, 1.1, "green"),
    (1, 0.01, 0.35, "blue"),
]

for y, x_lo, x_hi, color in bins:
    ax.barh(
        y,
        x_hi - x_lo,
        left=x_lo,
        height=0.6,
        color=color,
        edgecolor="black",
        linewidth=0.8,
    )

ax.barh(
    0,
    10.0 - 0.01,
    left=0.01,
    height=0.3,
    color="gray",
    alpha=0.6,
    edgecolor="black",
    linewidth=0.8,
)

y_labels = [
    "Runaway Greenhouse\n(Venus analog)",
    "Moist Greenhouse\n(stratospheric H2O loss)",
    "Stable Temperate\n(silicate weathering)",
    "Snowball\n(ice-albedo feedback)",
    "Gas Giant\n(no solid surface)",
]

annotations = [
    "Positive H2O-vapor feedback (Kasting 1988)",
    "Photodissociation + H escape (Kasting 1993)",
    "CO2-silicate thermostat (Walker 1981)",
    "Ice-albedo runaway (Hoffman 2002)",
    "H2-He envelope retention (Fulton 2017)",
]

for y, label, note in zip(range(4, -1, -1), y_labels, annotations):
    ax.text(
        0.009,
        y,
        label,
        ha="right",
        va="center",
        fontsize=11,
        fontweight="bold",
        transform=ax.get_yaxis_transform(),
        clip_on=False,
    )
    ax.text(
        1.02,
        y,
        note,
        ha="left",
        va="center",
        fontsize=9,
        fontstyle="italic",
        transform=ax.get_yaxis_transform(),
        clip_on=False,
    )

for x_edge in (0.35, 1.1, 1.5):
    ax.axvline(x_edge, linestyle="--", color="black", linewidth=1.2, alpha=0.8)

ax.set_xscale("log")
ax.set_xlim(0.01, 20)
ax.set_ylim(-0.5, 4.5)

ax.set_yticks([])
ax.set_xlabel("Stellar Insolation Flux S (S_earth)")
ax.grid(True, axis="x", which="both", alpha=0.25, linewidth=0.5)

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/fig5_climate_bins.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Figure 5 saved")
