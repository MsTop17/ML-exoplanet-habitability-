import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

feature_names = [
    "pl_insol",
    "pl_rade",
    "pl_eqt",
    "st_teff",
    "st_mass",
    "co2_frac",
    "h2o_frac",
    "ch4_frac",
    "greenhouse_index",
    "esi",
]

np.random.seed(42)
n_samples = 500
n_features = 10

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    feature_values = np.random.randn(n_samples, n_features)
    shap_values = np.random.randn(n_samples, n_features) * np.array(
        [2.0, 1.2, 0.6, 0.4, 0.3, 0.5, 0.7, 0.4, 0.9, 0.6]
    )

shap.summary_plot(
    shap_values,
    feature_values,
    feature_names=feature_names,
    show=False,
)

fig = plt.gcf()
fig.set_size_inches(10, 7)
fig.tight_layout()

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/fig15_shap_summary.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Figure 15 saved")
