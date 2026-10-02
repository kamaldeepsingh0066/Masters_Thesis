# ==========================================================
# NOTEBOOK 07 — UPDATED MERGE SECTION
# Add this AFTER your existing Notebook 07 merge code
# Merges ARIMA results with ML + LSTM for full comparison
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================================
# 1. LOAD ALL RESULT FILES
# ==========================================================

ml_results   = pd.read_csv("results/validation_results.csv")   # from Notebook 05
lstm_results = pd.read_csv("results/best_lstm_results.csv")     # from Notebook 07 existing
arima_results= pd.read_csv("results/arima_results.csv")         # from Notebook 08

# Clean LSTM (drop Epochs column if present)
if "Epochs" in lstm_results.columns:
    lstm_results = lstm_results.drop(columns=["Epochs"])

# ==========================================================
# 2. MERGE ALL INTO ONE MASTER RESULTS TABLE
# ==========================================================

all_results = pd.concat(
    [ml_results, lstm_results, arima_results],
    ignore_index=True
)

print(f"Total experiments in master results: {len(all_results)}")
print(f"Models: {sorted(all_results['Model'].unique())}")

# Save master
os.makedirs("results", exist_ok=True)
all_results.to_csv("results/all_results_final.csv", index=False)
print("Saved: results/all_results_final.csv")

# ==========================================================
# 3. OVERALL MODEL RANKING (avg across all targets/horizons)
# ==========================================================

model_ranking = (
    all_results
    .groupby("Model")
    .agg({"RMSE": "mean", "MAE": "mean", "R2": "mean"})
    .sort_values("RMSE")
    .round(2)
)
print("\nOverall Model Ranking:")
print(model_ranking)

# ==========================================================
# 4. HORIZON COMPARISON (all models averaged)
# ==========================================================

horizon_comparison = (
    all_results
    .groupby("Horizon")
    .agg({"RMSE": "mean", "MAE": "mean", "R2": "mean"})
    .sort_values("Horizon")
    .round(2)
)
print("\nHorizon Degradation:")
print(horizon_comparison)

# ==========================================================
# 5. ARIMA vs ML vs LSTM SUMMARY
# ==========================================================

# Group model families
def model_family(m):
    if m in ["ARIMA", "SARIMA", "SARIMAX"]:
        return "Statistical"
    elif m == "LSTM":
        return "Deep Learning"
    elif m == "Persistence":
        return "Baseline"
    else:
        return "Machine Learning"

all_results["Family"] = all_results["Model"].apply(model_family)

family_comparison = (
    all_results
    .groupby("Family")
    .agg({"RMSE": "mean", "MAE": "mean", "R2": "mean"})
    .sort_values("RMSE")
    .round(2)
)
print("\nModel Family Comparison (Statistical vs ML vs DL):")
print(family_comparison)

# ==========================================================
# 6. BEST MODEL PER TARGET × HORIZON
# (now includes ARIMA family)
# ==========================================================

best_configs = (
    all_results
    .loc[all_results.groupby(["Period", "Target", "Horizon"])["RMSE"].idxmin()]
    .sort_values(["Period", "Target", "Horizon"])
    .reset_index(drop=True)
)
print("\nBest Model per Period × Target × Horizon:")
print(best_configs[["Period","Target","Horizon","Model","RMSE","R2"]])

# Win count
win_count = (
    best_configs["Model"]
    .value_counts()
    .reset_index()
)
win_count.columns = ["Model", "Wins"]
print("\nModel Win Count:")
print(win_count)

# ==========================================================
# 7. PLOTS
# ==========================================================

# --- Plot 1: Overall RMSE by Model ---
fig, ax = plt.subplots(figsize=(10, 5))
model_ranking["RMSE"].plot(kind="bar", ax=ax, color="steelblue")
ax.set_title("Average RMSE by Model (All Targets & Horizons)")
ax.set_ylabel("Average RMSE (MWh)")
ax.set_xlabel("Model")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("results/plot_model_rmse_all.png", dpi=150)
plt.show()
print("Saved: results/plot_model_rmse_all.png")

# --- Plot 2: ARIMA vs ML vs LSTM by Horizon ---
fig, ax = plt.subplots(figsize=(10, 5))

# Pivot: model family × horizon
family_horizon = (
    all_results
    .groupby(["Family", "Horizon"])["RMSE"]
    .mean()
    .unstack("Family")
)
family_horizon.plot(ax=ax, marker="o")
ax.set_title("RMSE by Forecast Horizon — Statistical vs ML vs Deep Learning")
ax.set_xlabel("Forecast Horizon (hours)")
ax.set_ylabel("Average RMSE (MWh)")
ax.grid(True)
plt.tight_layout()
plt.savefig("results/plot_family_horizon_rmse.png", dpi=150)
plt.show()
print("Saved: results/plot_family_horizon_rmse.png")

# --- Plot 3: Per-Target comparison ---
for target in all_results["Target"].unique():
    subset = (
        all_results[all_results["Target"] == target]
        .groupby("Model")["RMSE"]
        .mean()
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    subset.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title(f"Average RMSE by Model — {target}")
    ax.set_ylabel("Average RMSE (MWh)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fname = f"results/plot_{target.lower()}_model_rmse.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")

# --- Plot 4: ARIMA family internal comparison ---
arima_only = (
    arima_results
    .groupby(["Model", "Horizon"])["RMSE"]
    .mean()
    .unstack("Model")
)
fig, ax = plt.subplots(figsize=(9, 5))
arima_only.plot(ax=ax, marker="o")
ax.set_title("ARIMA Family: RMSE by Horizon")
ax.set_xlabel("Forecast Horizon (hours)")
ax.set_ylabel("Average RMSE (MWh)")
ax.grid(True)
plt.tight_layout()
plt.savefig("results/plot_arima_family_horizon.png", dpi=150)
plt.show()
print("Saved: results/plot_arima_family_horizon.png")

print("\n" + "="*60)
print("FULL COMPARISON COMPLETE")
print(f"Total experiments across all models: {len(all_results)}")
print("="*60)
