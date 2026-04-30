Demand Forecasting and Inventory Optimization in Supply Chain Management
Project Overview
This project implements a comprehensive time series analysis pipeline to forecast product demand and optimize inventory levels. In supply chain management, accurate demand forecasting is essential to minimize the bullwhip effect, reduce holding costs, and prevent stockouts. This repository provides a structured approach to analyzing historical sales data, identifying patterns, and making prescriptive decisions for procurement and stock management.

Technical Workflow
The analysis follows a rigorous eight-step engineering pipeline:

Data Generation and Loading: Creation of a four-year monthly dataset featuring trends, seasonality, and holiday spikes.

Exploratory Data Analysis: Visualization of raw demand through rolling averages and monthly distribution box plots.

Seasonal Decomposition: Breaking down the time series into trend, seasonal, and residual components using additive modeling.

Stationarity Testing: Utilizing the Augmented Dickey-Fuller test to determine if the data requires differencing.

Correlation Analysis: Plotting Autocorrelation and Partial Autocorrelation Functions to identify parameters for statistical models.

Model Training: Implementation of Holt-Winters Triple Exponential Smoothing and SARIMA models.

Forecast Evaluation: Comparing models against a Seasonal Naive baseline using Mean Absolute Error and Mean Absolute Percentage Error.

Inventory Optimization: Applying forecast outputs to calculate Economic Order Quantity, Safety Stock, and Reorder Points.

Key Features
Dual Modeling Approach: Compares a state-space statistical model (SARIMA) with an exponential smoothing model (Holt-Winters).

Statistical Diagnostics: Includes formal testing for stationarity and lag correlations to justify model selection.

Prescriptive Analytics: Translates abstract forecasts into concrete business metrics such as Reorder Points.

Inventory Simulation: Includes a saw-tooth model visualization to illustrate inventory depletion and replenishment cycles.

Prerequisites
The following Python libraries are required to run this project:

Pandas: For data manipulation and time-series indexing.

NumPy: For mathematical operations and data simulation.

Statsmodels: For seasonal decomposition, ADF testing, and SARIMA modeling.

Matplotlib and Seaborn: For generating diagnostic and results plots.

Scikit-learn: For calculating error metrics.

Installation
To set up the environment, use the following commands:

git clone https://github.com/YourUsername/Project-Name.git
cd Project-Name
pip install pandas numpy statsmodels matplotlib seaborn scikit-learn

Usage
The main execution script contains the full pipeline. To run the analysis and generate the plots:

python demand_analysis.py

Model Evaluation Metrics
The performance of the forecasting models is evaluated based on:

Mean Absolute Error (MAE): Measures the average magnitude of errors in a set of forecasts.

Root Mean Squared Error (RMSE): Penalizes larger errors more heavily.

Mean Absolute Percentage Error (MAPE): Provides a percentage-based accuracy score for easier business interpretation.

Inventory Optimization Parameters
The project calculates three critical inventory metrics:

Safety Stock: Buffer inventory required to protect against demand variability.

Reorder Point (ROP): The inventory level at which a new replenishment order should be placed.

Economic Order Quantity (EOQ): The ideal order size that minimizes the total sum of ordering and holding costs.

License
This project is released under the MIT License. Details can be found in the LICENSE file.
