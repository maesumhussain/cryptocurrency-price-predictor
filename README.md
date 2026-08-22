# Regression Algorithms for Learning & Its Application on Cryptocurrency Markets

Syed Maesum Hussain Zaidi

## Project Overview

This project focuses on using regression based and other machine learning algorithms to predict cryptocurrency price movements. The system downloads historical market data from Yahoo Finance, engineers technical features, evaluates multiple models using walk-forward validation, and produces both next day price predictions and historical performance comparison between the algorithms.

The project supports both a console-based version and a graphical user interface (GUI).

The GUI version is the main implementation which allows the model to run with its full capabilities which includes being able to choose the cryptocurrency pairs, the years of data used and the choice between US Dollar and Great Britain Pound. 

The Console based version is the default version on which the GUI version is based. The console version only uses Bitcoin US Dollar with 10 year data for its price prediction and can not be edited. 

The algorithms included in the model are: 

- Ordinary Least Squares (OLS)
- Ridge Regression
- Lasso Regression
- Kernel Ridge Regression
- K-Nearest Neighbours (KNN)
- Feedforward Neural Network

The final runnable code for this project is located in:

`PROJECT/Programming Code/Final Code/`

## How It Works 

The model takes historical cryptocurrency data and learns patterns using different regression and other algorithms. It then uses these patterns to predict the next day’s closing price and evaluates how accurate each model is over time.

## Project Structure

The files in the project are:

- `config.py`: stores configuration settings and data classes
- `console_app.py`: runs the project in the console
- `crypto_pairs.py`: contains 25 handpicked cryptocurrency pairs
- `data_loader.py`: downloads and prepares historical data from Yahoo Finance
- `evaluation.py`: performs walk-forward evaluation across all algorithms
- `features.py`: creates the feature set used by the algorithms
- `gui_app.py`: launches the graphical user interface (GUI) used to run the model
- `gui_helpers.py`: helper functions for the GUI
- `gui_model_tabs.py`: model tab layout and image loading for the GUI
- `gui_performance_tab.py`: historical performance comparison tab in the GUI
- `gui_prediction_tab.py`: tomorrow prediction tab in the GUI
- `gui_styles.py`: GUI styling
- `metrics.py`: metric calculations
- `models_math.py`: mathematical implementations of the algorithms from scratch
- `models_wrappers.py`: wrapper classes used during tuning
- `plotting.py`: generates and saves plots
- `predictor.py`: generates tomorrow predictions
- `tuning.py`: hyperparameter tuning logic

- `tests/`: contains all the relevant test classes for the logic and other parts of the model code. 

## Evaluation Metrics

The models are evaluated using:
- Mean Squared Error (MSE) on Predicted Price
- Mean Squared Error (MSE) on Predicted Returns
- Directional Accuracy

## Required Imports and Libraries

This project uses the following Python imports.

### Standard library imports

- `os`
- `sys`
- `math`
- `datetime`
- `dataclasses`
- `typing`
- `zoneinfo`

### External libraries

- `numpy`
- `pandas`
- `matplotlib`
- `yfinance`
- `scikit-learn`
- `PyQt5`

## Running the Model

- First, you would need to make sure you have an internet connection because the model requires the data to be downloaded from Yahoo Finance to make the prediction and performance comparison.

- Secondly, you would clone the repository from git and then you would navigate to the correct folder `PROJECT/Programming Code/Final Code/`. 

- Thirdly, you would need to install the required external libraries which have been listed above. The command you need to run in your terminal to install these would look something like this:  
pip install numpy pandas matplotlib yfinance scikit-learn PyQt5

- Next, you would run the `gui_app.py` file in order to run the model. The command in the terminal to run this would look something like:
python gui_app.py

- Then, the model GUI will appear where you will select the cryptocurrency pair, the years of data to use (if that amount is not available the GUI will show you a message and automatically move it to the most amount available for that pair) and you would choose one of USD or GBP to see the results. Then you would click the button that says predict tomorrow's price and change and then the model will run. Once the run is complete then you can see the results. You can also navigate to the historical performance tab and then click on load historical performance to see the performance comparison of the algorithms and their ranking. You can also navigate to the specific tabs for each model where you can see the predicted vs actual prices and returns graph for each model and a one line description of them.

- Finally, There is alternate console implementation of the model which can run by running the `console_app.py` file. You would still need internet access and the same pre-requisites like installing the external libraries but this implementation will only run the set BTC/USD pair for 10 year data and give you the price predictions.

## Limitations

The model relies on historical price data and does not account for external factors such as news, market sentiment or macroeconomic events which can potentially affect prediction accuracy.
