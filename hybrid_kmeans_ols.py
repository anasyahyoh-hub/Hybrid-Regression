"""Hybrid K-Means + local polynomial OLS regression demonstration.

Run:
    python hybrid_kmeans_ols.py

Outputs are written to ./outputs.  The data set is synthetic so the example is
fully reproducible; replace make_dataset() with pd.read_csv(...) for real data.
"""

from pathlib import Path
import json
import os

# Keep Matplotlib's cache inside the project, including on restricted systems.
PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")  # non-interactive, portable image output
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")  # avoids platform core-detection warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

RANDOM_STATE = 42
N_CLUSTERS = 3
POLYNOMIAL_DEGREE = 2
OUT_DIR = PROJECT_DIR / "outputs"


def make_dataset(n_samples=450, random_state=RANDOM_STATE):
    """Make a noisy, piecewise non-linear 1D data set."""
    rng = np.random.default_rng(random_state)
    x = np.sort(rng.uniform(-6, 6, n_samples))
    y = np.piecewise(
        x,
        [x < -2, (x >= -2) & (x < 2), x >= 2],
        [
            lambda z: 0.55 * (z + 3.2) ** 2 - 3.6,
            lambda z: 0.35 * z**2 + 0.65 * z - 0.8,
            lambda z: -0.48 * (z - 3.4) ** 2 + 4.0,
        ],
    )
    y = y + rng.normal(0, 0.42, n_samples)
    return pd.DataFrame({"x": x, "y": y})


def build_ols(degree=POLYNOMIAL_DEGREE):
    """Polynomial basis followed by OLS (sklearn LinearRegression)."""
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("ols", LinearRegression()),
    ])


def predict_hybrid(x, kmeans, local_models):
    """Assign each x to its K-Means cluster and predict with that cluster's OLS."""
    labels = kmeans.predict(x.reshape(-1, 1))
    prediction = np.empty(len(x))
    for label, model in local_models.items():
        mask = labels == label
        prediction[mask] = model.predict(x[mask].reshape(-1, 1))
    return prediction, labels


def polynomial_equation(model):
    """Return readable coefficients: intercept + b1*x + b2*x^2 + ..."""
    ols = model.named_steps["ols"]
    return {"intercept": float(ols.intercept_),
            **{f"x^{i + 1}": float(value) for i, value in enumerate(ols.coef_)}}


def main():
    OUT_DIR.mkdir(exist_ok=True)
    data = make_dataset()
    train, test = train_test_split(data, test_size=0.30, random_state=RANDOM_STATE)
    x_train = train[["x"]].to_numpy()
    x_test = test[["x"]].to_numpy()
    y_train = train["y"].to_numpy()
    y_test = test["y"].to_numpy()

    # Cluster using the predictor X only, avoiding target leakage at prediction time.
    kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=20, random_state=RANDOM_STATE)
    train_labels = kmeans.fit_predict(x_train)

    local_models = {}
    local_rows = []
    for label in range(N_CLUSTERS):
        mask = train_labels == label
        model = build_ols().fit(x_train[mask], y_train[mask])
        local_models[label] = model
        local_rows.append({
            "cluster": label,
            "center_x": float(kmeans.cluster_centers_[label, 0]),
            "train_samples": int(mask.sum()),
            **polynomial_equation(model),
        })

    global_model = build_ols().fit(x_train, y_train)
    hybrid_pred, test_labels = predict_hybrid(x_test.ravel(), kmeans, local_models)
    global_pred = global_model.predict(x_test)
    metrics = pd.DataFrame([
        {"model": "Hybrid K-Means + Local OLS", "MSE": mean_squared_error(y_test, hybrid_pred),
         "RMSE": mean_squared_error(y_test, hybrid_pred) ** 0.5, "R2": r2_score(y_test, hybrid_pred)},
        {"model": "Global polynomial OLS", "MSE": mean_squared_error(y_test, global_pred),
         "RMSE": mean_squared_error(y_test, global_pred) ** 0.5, "R2": r2_score(y_test, global_pred)},
    ])
    metrics["MSE_improvement_vs_global_pct"] = [
        (1 - metrics.loc[0, "MSE"] / metrics.loc[1, "MSE"]) * 100, 0.0
    ]

    # Save complete tabular outputs.
    data.to_csv(OUT_DIR / "dataset.csv", index=False)
    pd.DataFrame(local_rows).sort_values("center_x").to_csv(OUT_DIR / "local_coefficients.csv", index=False)
    metrics.to_csv(OUT_DIR / "metrics.csv", index=False)
    with open(OUT_DIR / "results.json", "w", encoding="utf-8") as file:
        json.dump({"metrics": metrics.to_dict(orient="records"),
                   "global_coefficients": polynomial_equation(global_model)}, file, indent=2)

    # Plot local curves by predicted cluster across a dense x grid.
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    x_grid = np.linspace(data.x.min(), data.x.max(), 1000)
    grid_labels = kmeans.predict(x_grid.reshape(-1, 1))
    ax.scatter(x_train.ravel(), y_train, c=train_labels, cmap="tab10", s=22, alpha=.55,
               label="Training data (K-Means cluster)")
    ax.scatter(x_test.ravel(), y_test, c="black", s=11, alpha=.30, label="Test data")
    for label in range(N_CLUSTERS):
        mask = grid_labels == label
        order = np.argsort(x_grid[mask])
        x_part = x_grid[mask][order]
        y_part = local_models[label].predict(x_part.reshape(-1, 1))
        ax.plot(x_part, y_part, color=colors[label], lw=3,
                label=f"Local OLS: cluster {label}")
    ax.plot(x_grid, global_model.predict(x_grid.reshape(-1, 1)), "--", color="#d62728", lw=2.5,
            label="Global polynomial OLS")
    ax.set(title="Hybrid K-Means + Local OLS versus Global OLS", xlabel="x", ylabel="y")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "comparison_plot.png", dpi=180)
    plt.close(fig)

    print("\nModel comparison on held-out test data")
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nLocal polynomial coefficients (ordered by cluster center)")
    print(pd.DataFrame(local_rows).sort_values("center_x").to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nFiles saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
