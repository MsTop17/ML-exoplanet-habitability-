import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BIN_LABELS = ["Runaway GHG", "Moist GHG", "Stable Temp", "Snowball", "Gas Giant"]
BIN_COLORS = {0: "red", 1: "orange", 2: "green", 3: "blue", 4: "gray"}


def assign_bin(pl_rade, pl_insol):
    if pl_rade >= 1.6:
        return 4
    if pl_insol > 1.5:
        return 0
    if pl_insol > 1.1:
        return 1
    if pl_insol >= 0.35:
        return 2
    return 3


def load_data():
    for path in ("data/processed/labeled_data.csv", "data/processed/clean_data.csv"):
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except (OSError, pd.errors.ParserError) as exc:
                print(f"Could not read {path}: {exc}")

    print("No processed data found; generating synthetic data")
    rng = np.random.default_rng(42)
    n = 1000
    n_giant = 800
    rocky_insol = np.concatenate(
        [
            rng.uniform(1.5, 4.0, 120),
            rng.uniform(1.1, 1.5, 30),
            rng.uniform(0.35, 1.1, 40),
            rng.uniform(1e-3, 0.35, n - n_giant - 120 - 30 - 40),
        ]
    )
    pl_insol = np.concatenate([rocky_insol, rng.uniform(0.01, 20.0, n_giant)])
    pl_rade = np.concatenate(
        [rng.uniform(0.3, 1.6, len(rocky_insol)), rng.uniform(1.6, 15.0, n_giant)]
    )
    order = rng.permutation(n)
    return pd.DataFrame({"pl_rade": pl_rade[order], "pl_insol": pl_insol[order]})


df = load_data()

if "climate_bin" not in df.columns:
    df["climate_bin"] = [
        assign_bin(r, s) for r, s in zip(df["pl_rade"], df["pl_insol"])
    ]

df = df.dropna(subset=["pl_insol", "pl_rade"])
print(f"Plotting {len(df)} planets")

x = np.log10(df["pl_insol"].clip(lower=1e-4))
y = np.log10(df["pl_rade"].clip(lower=1e-4))
bins = df["climate_bin"].astype(int)

fig, ax = plt.subplots(figsize=(10, 7))

for b, label in enumerate(BIN_LABELS):
    mask = bins == b
    ax.scatter(
        x[mask],
        y[mask],
        c=BIN_COLORS[b],
        label=label,
        alpha=0.5,
        s=15,
        edgecolors="none",
    )

for edge in (0.35, 1.1, 1.5):
    ax.axvline(np.log10(edge), linestyle="--", color="black", linewidth=1.2, alpha=0.8)

ax.axhline(np.log10(1.6), linestyle="--", color="black", linewidth=1.2)
ax.text(
    x.max() * 0.98,
    np.log10(1.6) + 0.03,
    "Fulton gap",
    ha="right",
    va="bottom",
    fontsize=10,
    color="black",
)

for x_s, y_s, name, offset in [
    (0.0, 0.0, "Earth", (10, 10)),
    (np.log10(0.43), np.log10(0.53), "Mars", (10, 10)),
    (np.log10(1.9), np.log10(0.95), "Venus", (10, 10)),
]:
    ax.plot(
        x_s,
        y_s,
        marker="*",
        markersize=16,
        color="black",
        markeredgecolor="white",
        markeredgewidth=0.8,
        linestyle="none",
    )
    ax.annotate(
        name,
        xy=(x_s, y_s),
        xytext=offset,
        textcoords="offset points",
        fontsize=11,
        color="black",
    )

ax.set_xlabel("log10(Insolation Flux S/S_earth)")
ax.set_ylabel("log10(Radius R/R_earth)")
ax.grid(True, which="both", alpha=0.2, linewidth=0.5)
ax.legend(loc="upper right", fontsize=9, markerscale=2)

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/fig10_scatter_hz.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Figure 10 saved")
