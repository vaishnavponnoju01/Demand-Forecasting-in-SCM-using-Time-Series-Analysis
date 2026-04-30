import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
from fpdf import FPDF
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- PDF REPORT ENGINE ---
class SCMReport(FPDF):
    def header(self):
        if self.page_no() == 1:
            self.set_font('Helvetica', 'B', 20)
            self.cell(0, 15, 'Demand Forecasting & Inventory Optimization', 0, 1, 'C')
            self.set_font('Helvetica', 'I', 10)
            self.cell(0, 10, 'B.Tech Project Technical Report - Supply Chain Analytics', 0, 1, 'C')
            self.ln(5)

    def add_section_header(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.set_fill_color(240, 240, 240)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(3)

    def add_concept(self, text):
        self.set_font('Helvetica', 'I', 10)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 6, f"Concept: {text}")
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def add_analysis(self, text, style=''):
        if style.lower() == 'courier':
            self.set_font('Courier', '', 9)
        else:
            self.set_font('Helvetica', '', 11)
        self.multi_cell(0, 7, text)
        self.ln(2)

    def add_plot(self, path, caption):
        self.image(path, x=25, w=160)
        self.set_font('Helvetica', 'I', 9)
        self.cell(0, 10, f"Figure: {caption}", 0, 1, 'C')
        self.ln(5)

# --- INITIALIZATION ---
warnings.filterwarnings("ignore")
np.random.seed(42)
sns.set_style("whitegrid")
report = SCMReport()
report.add_page()

ASSETS_DIR = "report_assets"
if os.path.exists(ASSETS_DIR): shutil.rmtree(ASSETS_DIR)
os.makedirs(ASSETS_DIR)

# ---------------------------------------------------------
# STEP 1: DATA GENERATION
# ---------------------------------------------------------
report.add_section_header("1. Data Acquisition and Characteristics")
report.add_concept("Demand in supply chains is a composite of a base level, long-term trends, and seasonal cycles. Real-world data often contains noise and holiday-related spikes.")

dates = pd.date_range(start="2021-01-01", periods=48, freq="MS")
t = np.arange(48)
demand = (300 + 5 * t + 80 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 25, 48))
for i, d in enumerate(dates):
    if d.month in [11, 12]: demand[i] += 120
df = pd.DataFrame({"Demand": demand.astype(int)}, index=dates)

report.add_analysis(f"Dataset generated for 48 months (2021-2024). \nInitial Demand Level: 300 units. \nGrowth Trend: 5 units/month.")

# ---------------------------------------------------------
# STEP 2: VISUALIZATION
# ---------------------------------------------------------
report.add_section_header("2. Exploratory Data Analysis (EDA)")
report.add_concept("Moving averages act as a low-pass filter to remove short-term noise and reveal the underlying direction of the supply chain. Boxplots identify periodic seasonal variance.")

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
axes[0].plot(df.index, df["Demand"], color="steelblue", label="Raw Demand")
axes[0].plot(df["Demand"].rolling(3).mean(), color="orange", label="3-Month Smoothing", linewidth=2)
axes[0].legend()
df["Month_Name"] = df.index.strftime("%b")
sns.boxplot(data=df, x="Month_Name", y="Demand", palette="Blues", ax=axes[1])
plt.tight_layout()
path2 = f"{ASSETS_DIR}/step2.png"
plt.savefig(path2)
report.add_plot(path2, "Demand Trends and Monthly Distribution Analysis")

# ---------------------------------------------------------
# STEP 3: DECOMPOSITION
# ---------------------------------------------------------
report.add_section_header("3. Time Series Decomposition")
report.add_concept("Decomposition deconstructs demand into Trend, Seasonal, and Residual components. This allows managers to distinguish between permanent growth and temporary seasonal spikes.")

decomp = seasonal_decompose(df["Demand"], model="additive", period=12)
fig = decomp.plot()
fig.set_size_inches(12, 8)
plt.tight_layout()
path3 = f"{ASSETS_DIR}/step3.png"
plt.savefig(path3)
report.add_page()
report.add_plot(path3, "Additive Decomposition showing isolated Trend and Seasonality")

# ---------------------------------------------------------
# STEP 4: STATIONARITY (ADF)
# ---------------------------------------------------------
report.add_section_header("4. Statistical Stationarity Testing")
report.add_concept("Stationarity implies that mean and variance are constant over time. If a series is non-stationary, models like SARIMA cannot be reliably applied without differencing.")

res = adfuller(df["Demand"])
adf_text = f"Augmented Dickey-Fuller Test Results:\nADF Statistic: {res[0]:.4f}\np-value: {res[1]:.4f}\nConclusion: {'Stationary' if res[1] < 0.05 else 'Non-Stationary'}"
report.add_analysis(adf_text)

# ---------------------------------------------------------
# STEP 5: ACF & PACF
# ---------------------------------------------------------
report.add_section_header("5. Correlation Analysis (ACF & PACF)")
report.add_concept("Autocorrelation (ACF) identifies the 'memory' of the demand, while Partial Autocorrelation (PACF) determines the direct relationship between lags, used to set SARIMA orders (p, q).")

series_diff = df["Demand"].diff().dropna()
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(series_diff, lags=20, ax=axes[0])
plot_pacf(series_diff, lags=20, ax=axes[1])
path5 = f"{ASSETS_DIR}/step5.png"
plt.savefig(path5)
report.add_plot(path5, "ACF and PACF Plots of Differenced Demand Data")

# ---------------------------------------------------------
# STEP 6 & 7: MODELING & RESULTS
# ---------------------------------------------------------
report.add_page()
report.add_section_header("6. Forecasting Model Performance")
report.add_concept("We compare Holt-Winters (Exponential Smoothing) against SARIMA (Stochastic Modeling). The Mean Absolute Percentage Error (MAPE) serves as the primary accuracy metric.")

train, test = df["Demand"].iloc[:-12], df["Demand"].iloc[-12:]
hw_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit()
hw_fc = hw_model.forecast(12)
sarima_model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
sarima_fc = sarima_model.forecast(12)

def get_metrics(actual, pred):
    mae = mean_absolute_error(actual, pred)
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    return mae, mape

m_hw = get_metrics(test, hw_fc)
m_s = get_metrics(test, sarima_fc)

res_table = f"{'Model Name':<20} | {'MAE':<10} | {'MAPE (%)':<10}\n"
res_table += "-"*45 + "\n"
res_table += f"{'Holt-Winters':<20} | {m_hw[0]:<10.1f} | {m_hw[1]:<10.2f}%\n"
res_table += f"{'SARIMA':<20} | {m_s[0]:<10.1f} | {m_s[1]:<10.2f}%"
report.add_analysis(res_table, 'courier')

plt.figure(figsize=(12, 5))
plt.plot(df["Demand"], label="Actual Demand", color="black", alpha=0.4)
plt.plot(hw_fc, label="Holt-Winters", linestyle="--")
plt.plot(sarima_fc, label="SARIMA", linestyle="--")
plt.legend()
path7 = f"{ASSETS_DIR}/step7.png"
plt.savefig(path7)
report.add_plot(path7, "Comparative Forecast Visualization")

# ---------------------------------------------------------
# STEP 8: INVENTORY OPTIMIZATION
# ---------------------------------------------------------
report.add_section_header("7. Inventory Strategy Optimization")
report.add_concept("Safety Stock provides a buffer against forecast error. EOQ balances ordering costs against holding costs. The Reorder Point (ROP) triggers replenishment based on lead time demand.")

best_fc = sarima_fc if m_s[1] < m_hw[1] else hw_fc
avg_d, std_d = best_fc.mean(), df["Demand"].std()
z, L, holding, ordering = 1.96, 1, 10, 500

ss = z * std_d * np.sqrt(L)
rop = (avg_d * L) + ss
eoq = np.sqrt((2 * avg_d * 12 * ordering) / holding)

inv_out = f"Safety Stock: {int(ss)} units\nReorder Point (ROP): {int(rop)} units\nEconomic Order Quantity (EOQ): {int(eoq)} units"
report.add_analysis(inv_out)

# Saw-tooth diagram
cycle = int(eoq / avg_d * 30)
t_sim = np.linspace(0, 2 * cycle, 200)
inv_sim = [ss + eoq * (1 - (tp % cycle)/cycle) for tp in t_sim]
plt.figure(figsize=(10, 4))
plt.plot(t_sim, inv_sim, label="Inventory Level", color="darkcyan")
plt.axhline(ss, color="red", linestyle="--", label="Safety Stock")
plt.title("Inventory Replenishment Simulation (Saw-Tooth Model)")
plt.legend()
path8 = f"{ASSETS_DIR}/step8.png"
plt.savefig(path8)
report.add_plot(path8, "Inventory Level Simulation over Time")

# Final Export
report.output("Demand_Forecasting_Technical_Report.pdf")
shutil.rmtree(ASSETS_DIR)
print("\nFinal Technical Report Generated: Demand_Forecasting_Technical_Report.pdf")