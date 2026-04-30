"""
Demand Forecasting in Supply Chain Management
Using Time Series Analysis
====================================================================
Tools : Python, Pandas, Statsmodels, Matplotlib
B.Tech Project


FLOW
----
Step 1 : Generate / Load Data
Step 2 : Visualize the Time Series
Step 3 : Decompose (Trend + Seasonal + Residual)
Step 4 : Check Stationarity (ADF Test)
Step 5 : ACF & PACF Plots  →  choose SARIMA order
Step 6 : Train Models (Holt-Winters & SARIMA)
Step 7 : Forecast & Compare Accuracy
Step 8 : Inventory Optimization (Safety Stock, EOQ, ROP)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.seasonal         import seasonal_decompose
from statsmodels.tsa.stattools        import adfuller
from statsmodels.graphics.tsaplots    import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters      import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
np.random.seed(42)

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 110, "axes.titlesize": 13,
"axes.labelsize": 11, "legend.fontsize": 9})



# STEP 1 — GENERATE DATA
# (In a real project, replace this with:  df = pd.read_csv("data.csv"))


# Simulate 4 years of monthly demand data
dates  = pd.date_range(start="2021-01-01", periods=48, freq="MS")
t      = np.arange(48)

# Demand = Base + Linear Trend + Seasonality + Random Noise
demand = (300
          + 5 * t                                        # upward trend
          + 80 * np.sin(2 * np.pi * t / 12)             # yearly seasonality
+ np.random.normal(0, 25, 48))                 # noise

# Holiday spike in November & December (Diwali / Christmas)
for i, d in enumerate(dates):
    if d.month in [11, 12]:
        demand[i] += 120

df = pd.DataFrame({"Demand": demand.astype(int)}, index=dates)
df.index.name = "Month"

print("Dataset Preview:")
print(df.head(12))
print(f"\nTotal Months : {len(df)}")
print(f"Demand Range : {df['Demand'].min()} – {df['Demand'].max()} units")



# STEP 2 — VISUALIZE THE TIME SERIES


fig, axes = plt.subplots(2, 1, figsize=(13, 8))
fig.suptitle("Step 2 — Time Series Visualization", fontsize=14, fontweight="bold")

# --- Raw series + 3-month moving average ---
axes[0].plot(df.index, df["Demand"], color="steelblue",
linewidth=1.5, label="Monthly Demand", alpha=0.8)
axes[0].plot(df["Demand"].rolling(3).mean(), color="darkorange",
linewidth=2.2, label="3-Month Moving Average")
axes[0].set_title("Monthly Demand with Moving Average")
axes[0].set_ylabel("Demand (units)")
axes[0].legend()

# --- Monthly box plot (shows seasonality) ---
df["Month_Name"] = df.index.strftime("%b")
month_order = ["Jan","Feb","Mar","Apr","May","Jun",
"Jul","Aug","Sep","Oct","Nov","Dec"]
sns.boxplot(data=df, x="Month_Name", y="Demand",
            order=month_order, palette="Blues", ax=axes[1])
axes[1].set_title("Seasonal Pattern — Demand Distribution by Month")
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Demand (units)")

plt.tight_layout()
plt.show()


# STEP 3 — DECOMPOSITION
# Break the series into: Trend + Seasonal + Residual


print("\n" + "="*55)
print("Step 3 — Seasonal Decomposition")
print("="*55)

decomp = seasonal_decompose(df["Demand"], model="additive", period=12)

fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
fig.suptitle("Step 3 — Additive Decomposition\n"
"(Observed = Trend + Seasonal + Residual)",
fontsize=13, fontweight="bold")

components = [
    (df["Demand"],    "Observed",  "steelblue"),
    (decomp.trend,   "Trend",     "darkorange"),
    (decomp.seasonal,"Seasonal",  "green"),
    (decomp.resid,   "Residual",  "red"),
]
for ax, (data, label, color) in zip(axes, components):
    if label == "Residual":
        ax.bar(data.dropna().index, data.dropna(), color=color, alpha=0.6, width=20)
    else:
        ax.plot(data, color=color, linewidth=1.6)
    ax.set_ylabel(label)
    ax.axhline(0, linestyle="--", color="grey", linewidth=0.7)

plt.tight_layout()
plt.show()



# STEP 4 — STATIONARITY TEST (Augmented Dickey-Fuller)
# A time series MUST be stationary before applying ARIMA models.
# Stationarity = constant mean, variance, and no seasonality.


print("\n" + "="*55)
print("Step 4 — Stationarity Test (ADF Test)")
print("="*55)

def adf_test(series, label="Series"):
    result = adfuller(series.dropna(), autolag="AIC")
    adf_stat, p_value = result[0], result[1]
    print(f"\n  Series       : {label}")
    print(f"  ADF Statistic: {adf_stat:.4f}")
    print(f"  p-value      : {p_value:.4f}")
    if p_value <= 0.05:
        print("  Result  ✓  STATIONARY  (p ≤ 0.05) — safe to model")
    else:
        print("  Result  ✗  NON-STATIONARY  (p > 0.05) — differencing needed")
    return p_value

# Test raw series
p_raw = adf_test(df["Demand"], "Raw Demand")

# If non-stationary, apply first-order differencing and retest
if p_raw > 0.05:
    df["Demand_diff"] = df["Demand"].diff()
    adf_test(df["Demand_diff"].dropna(), "First-Order Differenced")

print("""
Interpretation:
    d = 0  → already stationary
    d = 1  → first difference makes it stationary  ← our case
    D = 1  → seasonal difference (lag 12)
""")



# STEP 5 — ACF & PACF PLOTS
# ACF  → tells us the MA (q) order
# PACF → tells us the AR (p) order
# Spikes at lag 12, 24 → seasonal component present


print("\n" + "="*55)
print("Step 5 — ACF & PACF Analysis")
print("="*55)

series_diff = df["Demand"].diff().dropna()

fig, axes = plt.subplots(2, 2, figsize=(13, 7))
fig.suptitle("Step 5 — ACF & PACF (Model Order Identification)",
fontsize=13, fontweight="bold")

max_lags = min(24, len(series_diff) // 2 - 1)

plot_acf (df["Demand"],  lags=min(24, len(df)-1), ax=axes[0][0],
color="steelblue",   title="ACF — Original Series")
plot_pacf(df["Demand"],  lags=min(24, len(df)//2-1), ax=axes[0][1],
color="steelblue",   title="PACF — Original Series", method="ywm")
plot_acf (series_diff,   lags=max_lags, ax=axes[1][0],
color="darkorange",  title="ACF — After Differencing (d=1)")
plot_pacf(series_diff,   lags=max_lags, ax=axes[1][1],
color="darkorange",  title="PACF — After Differencing (d=1)", method="ywm")

for ax in axes.flat:
    ax.axvline(12, color="red", linestyle=":", linewidth=1.2,
alpha=0.7, label="Lag 12")
    ax.set_xlabel("Lags (months)")

fig.text(0.01, 0.01,
"Reading:  ACF cuts off at lag q → MA(q) order  |  "
"PACF cuts off at lag p → AR(p) order  |  "
"Spikes at 12 → seasonal period",
fontsize=8, color="dimgrey")

plt.tight_layout()
plt.show()

print("""
From the plots:
    p = 1  (PACF cuts off after lag 1)
    d = 1  (one difference needed)
    q = 1  (ACF cuts off after lag 1)
    Seasonal: P=1, D=1, Q=1, s=12
→ SARIMA(1,1,1)(1,1,1)[12]
""")



# STEP 6 — TRAIN / TEST SPLIT & MODEL TRAINING
# Train on 3 years, test on last 12 months


print("\n" + "="*55)
print("Step 6 — Model Training")
print("="*55)

train = df["Demand"].iloc[:-12]   # first 36 months
test  = df["Demand"].iloc[-12:]   # last 12 months

print(f"  Training months : {len(train)}  ({train.index[0].strftime('%b %Y')} – {train.index[-1].strftime('%b %Y')})")
print(f"  Testing months  : {len(test)}   ({test.index[0].strftime('%b %Y')} – {test.index[-1].strftime('%b %Y')})")

# --- Model A: Holt-Winters Triple Exponential Smoothing ---
print("\n  Training Holt-Winters (Triple Exponential Smoothing)...")
hw_model    = ExponentialSmoothing(train, trend="add",
seasonal="add", seasonal_periods=12).fit()
hw_forecast = hw_model.forecast(12)

# --- Model B: SARIMA ---
print("  Training SARIMA(1,1,1)(1,1,1)[12]...")
sarima_model    = SARIMAX(train, order=(1,1,1),
seasonal_order=(1,1,1,12)).fit(disp=False)
sarima_forecast = sarima_model.forecast(12)

# --- Baseline: Seasonal Naïve (repeat last year) ---
naive_forecast = pd.Series(train.values[-12:], index=test.index)

print("  Done.")


# ====================================================================
# STEP 7 — FORECAST RESULTS & ACCURACY METRICS
# ====================================================================

print("\n" + "="*55)
print("Step 7 — Forecast Comparison & Accuracy")
print("="*55)

def accuracy(actual, predicted):
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mae, rmse, mape

mae_hw, rmse_hw, mape_hw         = accuracy(test, hw_forecast)
mae_s,  rmse_s,  mape_s          = accuracy(test, sarima_forecast)
mae_n,  rmse_n,  mape_n          = accuracy(test, naive_forecast)

print(f"\n  {'Model':<25} {'MAE':>8} {'RMSE':>8} {'MAPE':>8} {'Accuracy':>10}")
print("  " + "-"*60)
print(f"  {'Holt-Winters':<25} {mae_hw:>8.1f} {rmse_hw:>8.1f} {mape_hw:>7.2f}% {100-mape_hw:>9.2f}%")
print(f"  {'SARIMA(1,1,1)(1,1,1)[12]':<25} {mae_s:>8.1f} {rmse_s:>8.1f} {mape_s:>7.2f}% {100-mape_s:>9.2f}%")
print(f"  {'Seasonal Naïve (baseline)':<25} {mae_n:>8.1f} {rmse_n:>8.1f} {mape_n:>7.2f}% {100-mape_n:>9.2f}%")

# --- Plot: Actuals vs Forecasts ---
plt.figure(figsize=(13, 6))
plt.plot(train.index, train,           color="steelblue",  linewidth=1.8,  label="Training Data")
plt.plot(test.index,  test,            color="black",       linewidth=2,    label="Actual Demand", zorder=5)
plt.plot(test.index,  hw_forecast,     color="crimson",     linewidth=2, linestyle="--", label=f"Holt-Winters  (Acc: {100-mape_hw:.1f}%)")
plt.plot(test.index,  sarima_forecast, color="green",       linewidth=2, linestyle="--", label=f"SARIMA         (Acc: {100-mape_s:.1f}%)")
plt.plot(test.index,  naive_forecast,  color="orange",      linewidth=1.5, linestyle=":", label=f"Seasonal Naïve (Acc: {100-mape_n:.1f}%)")
plt.axvline(test.index[0], color="grey", linestyle=":", linewidth=1.2)
plt.text(test.index[0], plt.ylim()[0] + 10, " Test Period →",
color="grey", fontsize=9)
plt.title("Step 7 — 12-Month Demand Forecast vs Actual",
fontsize=13, fontweight="bold")
plt.ylabel("Demand (units)")
plt.xlabel("Month")
plt.legend()
plt.tight_layout()
plt.show()


# --- Bar chart: Model accuracy comparison ---
models  = ["Holt-Winters", "SARIMA", "Seasonal Naïve"]
mapes   = [mape_hw, mape_s, mape_n]
colors  = ["crimson", "green", "orange"]

plt.figure(figsize=(7, 5))
bars = plt.bar(models, mapes, color=colors, width=0.4, edgecolor="white")
for bar, val in zip(bars, mapes):
    plt.text(bar.get_x() + bar.get_width()/2,
bar.get_height() + 0.2,
        f"{val:.2f}%", ha="center", fontweight="bold", fontsize=10)
plt.title("Model Comparison — MAPE (Lower is Better)",
fontsize=12, fontweight="bold")
plt.ylabel("MAPE (%)")
plt.ylim(0, max(mapes) * 1.3)
plt.tight_layout()
plt.show()


# ====================================================================
# STEP 8 — INVENTORY OPTIMIZATION
# Using the best forecast to compute:
#   Safety Stock — buffer against demand uncertainty
#   Reorder Point — when to place a new order
#   EOQ — how much to order at once
# ====================================================================

print("\n" + "="*55)
print("Step 8 — Inventory Optimization")
print("="*55)

# Pick best model by MAPE
best_forecast = sarima_forecast if mape_s < mape_hw else hw_forecast
best_name     = "SARIMA" if mape_s < mape_hw else "Holt-Winters"
print(f"\n  Using forecast from: {best_name}")

# Parameters
avg_demand   = best_forecast.mean()      # avg monthly demand
std_demand   = df["Demand"].std()        # historical variability
lead_time    = 1                          # months
z            = 1.96                       # 97.5% service level
holding_cost = 10                         # ₹ per unit per month
order_cost   = 500                        # ₹ per order

# Calculations
safety_stock = z * std_demand * np.sqrt(lead_time)
rop          = (avg_demand * lead_time) + safety_stock
eoq          = np.sqrt((2 * avg_demand * 12 * order_cost) / holding_cost)

print(f"""
Input Parameters
─────────────────────────────────────────
Avg Monthly Demand (forecast) : {avg_demand:>8.0f} units
Demand Std Dev (historical)   : {std_demand:>8.1f} units
Lead Time                     : {lead_time:>8} month(s)
Service Level (z = {z})      : {97.5:>8}%
Holding Cost                  : ₹{holding_cost:>7}/unit/month
Ordering Cost                 : ₹{order_cost:>7}/order

Optimization Results
─────────────────────────────────────────
Safety Stock    : {int(safety_stock):>6} units
Reorder Point   : {int(rop):>6} units
Economic Order Qty (EOQ) : {int(eoq):>6} units
""")

# --- Inventory level diagram ---
cycle_len = int(eoq / avg_demand * 30)   # scaled for display
t_plot    = np.linspace(0, 3 * cycle_len, 300)
inventory = []
for tp in t_plot:
    pos = tp % cycle_len
    inventory.append(safety_stock + eoq * (1 - pos / cycle_len))

plt.figure(figsize=(10, 5))
plt.plot(t_plot, inventory, color="steelblue", linewidth=2, label="Inventory Level")
plt.axhline(safety_stock, color="red",    linestyle="--", linewidth=1.5,
            label=f"Safety Stock = {int(safety_stock)} units")
plt.axhline(rop,          color="orange", linestyle=":",  linewidth=1.5,
            label=f"Reorder Point = {int(rop)} units")
plt.fill_between(t_plot, 0, safety_stock, color="red",  alpha=0.08)
plt.fill_between(t_plot, safety_stock, inventory,
    color="steelblue", alpha=0.12, label="Cycle Stock")
plt.title("Step 8 — Inventory Level over Time (Saw-Tooth Model)",
fontsize=13, fontweight="bold")
plt.ylabel("Inventory (units)")
plt.xlabel("Time →")
plt.legend()
plt.tight_layout()
plt.show()

print("="*55)
print("  Project Complete!")
print("="*55)
