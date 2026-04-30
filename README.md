# Demand Forecasting and Inventory Optimization in Supply Chain Management

## Project Description

This project implements an end to end pipeline for time series analysis to forecast product demand and determine optimal inventory levels. It utilizes statistical methods to identify trends and seasonality in historical sales data, providing a scientific basis for supply chain decision making. The pipeline includes data visualization, stationarity testing, model training using SARIMA and Holt Winters, and inventory metrics calculation.

## Requirements

The project is built using Python 3 and requires the following libraries:

* pandas
* numpy
* statsmodels
* matplotlib
* seaborn
* scikit-learn
* fpdf

## Installation and Setup

Follow these steps to set up the project environment on your local machine.

### Step 1: Create a Project Folder

Open your terminal or command prompt and run the following commands to create and enter a new directory:

mkdir Demand-Forecasting-Project
cd Demand-Forecasting-Project

### Step 2: Set Up a Virtual Environment

It is recommended to use a virtual environment to manage dependencies. Run the appropriate command for your operating system.

python -m venv venv

To activate the environment on Windows:
.\venv\Scripts\activate

To activate the environment on macOS or Linux:
source venv/bin/activate

### Step 3: Install Required Libraries

Once the virtual environment is active, install the necessary packages using pip:

pip install pandas numpy statsmodels matplotlib seaborn scikit-learn fpdf

## Execution

After the installation is complete, ensure your main script is named DemandforecastTSanalysis.py and run it using:

python DemandforecastTSanalysis.py

## Technical Workflow

The analysis follows a rigorous eight-step engineering pipeline:

1. Data Generation and Loading: Creation of a four-year monthly dataset featuring trends, seasonality, and holiday spikes.
2. Exploratory Data Analysis: Visualization of raw demand through rolling averages and monthly distribution box plots.
3. Seasonal Decomposition: Breaking down the time series into trend, seasonal, and residual components using additive modeling.
4. Stationarity Testing: Utilizing the Augmented Dickey-Fuller test to determine if the data requires differencing.
5. Correlation Analysis: Plotting Autocorrelation and Partial Autocorrelation Functions to identify parameters for statistical models.
6. Model Training: Implementation of Holt-Winters Triple Exponential Smoothing and SARIMA models.
7. Forecast Evaluation: Comparing models against a Seasonal Naive baseline using Mean Absolute Error and Mean Absolute Percentage Error.
8. Inventory Optimization: Applying forecast outputs to calculate Economic Order Quantity, Safety Stock, and Reorder Points.

## Key Features

- Dual Modeling Approach: Compares a state-space statistical model (SARIMA) with an exponential smoothing model (Holt-Winters).
- Statistical Diagnostics: Includes formal testing for stationarity and lag correlations to justify model selection.
- Prescriptive Analytics: Translates abstract forecasts into concrete business metrics such as Reorder Points.
- Inventory Simulation: Includes a saw-tooth model visualization to illustrate inventory depletion and replenishment cycles.

## Inventory Optimization Results

The system outputs actionable metrics based on the most accurate forecast:

* Safety Stock: The buffer quantity to prevent stockouts during demand spikes.
* Reorder Point: The specific inventory level that triggers a new replenishment order.
* Economic Order Quantity: The optimal order size to minimize total ordering and holding costs.
## Tech Stack

- Language: Python 3.x
- Libraries: Pandas, NumPy, Statsmodels, Scikit-learn, Matplotlib, Seaborn
## Model Evaluation Metrics

The performance of the forecasting models is evaluated based on:

- Mean Absolute Error (MAE): Measures the average magnitude of errors in a set of forecasts.
- Root Mean Squared Error (RMSE): Penalizes larger errors more heavily.
- Mean Absolute Percentage Error (MAPE): Provides a percentage-based accuracy score for easier business interpretation.

## License
This project is licensed under the MIT License.











