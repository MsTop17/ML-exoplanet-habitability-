import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge

fig, ax = plt.subplots(figsize=(9, 9))

# Shaded zones (drawn first so orbits and bodies sit on top)
ax.add_patch(Circle((0, 0), 0.95, color="red", alpha=0.25, zorder=1))
ax.add_patch(Wedge((0, 0), 1.55, 0, 360, width=1.55 - 0.95, color="green", alpha=0.35, zorder=1))
ax.add_patch(Circle((0, 0), 4.0, color="lightblue", alpha=0.20, zorder=0))

# Dashed reference orbits
orbit_radii = [1.0, 1.5, 2.2, 3.5]
for r in orbit_radii:
    ax.add_patch(
        Circle((0, 0), r, fill=False, linestyle="--", color="gray", linewidth=1, zorder=2)
    )

# Central star
ax.add_patch(Circle((0, 0), 0.15, facecolor="gold", edgecolor="orange", zorder=4))

# Planets
planets = [
    ((0.7, 0.0), "orange", "Venus analog"),
    ((1.0, 0.9), "blue", "Earth"),
    ((1.5, -0.7), "red", "Mars analog"),
    ((2.2, 1.2), "cyan", "Outer orbit"),
]
for (px, py), color, label in planets:
    ax.plot(px, py, "o", color=color, markersize=9, zorder=5)
    ax.annotate(
        label,
        xy=(px, py),
        xytext=(10, 8),
        textcoords="offset points",
        fontsize=9,
        color="black",
        zorder=5,
    )

# Region labels
region_labels = [
    ((0.4, 0.4), "Too Hot\n(Runaway Greenhouse)"),
    ((1.25, -1.3), "Habitable Zone"),
    ((2.8, -0.3), "Too Cold\n(Snowball)"),
]
for (tx, ty), text in region_labels:
    ax.text(tx, ty, text, fontsize=11, fontweight="bold", ha="center", va="center", zorder=5)

ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_aspect("equal")
ax.axis("off")

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/fig0_hz_schematic.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("First Project Image saved")
