# Meta-learning and data Augmentation to Stress Test Forecasting Models (MAST)

Repository complementary to the paper `MAST`: Meta-learning and data Augmentation to Stress Test Forecasting Models

## Usage

To use the classes and scripts provided in this repository, follow the instructions outlined in the documentation and code comments. Make sure to install the required dependencies listed in the `requirements.txt` file before running any scripts.

To initiate experiments and evaluate different models and resampling techniques, use `main.py`. Specify the scripts and resamplers to use within `main.py` before execution.

## Classes Overview

This repository includes several Python classes designed for different stages of the metamodel creation for predicting and evaluating stress testing and conditions:

- `MAST`: Class for model evaluation and feature extraction.
- `ForecastingModel`: Class for training, tuning and forecasting using a LightGBM model.
- `BaselineModel`: Class for training and forecasting using a baseline model.
- `MetaModel`: Class for training a metamodel using LightGBM with optional data resampling.
- `PrepareDataset`: Class for loading and preprocessing datasets.
- `Holdout`: Class for hold-out cross-validator generator.

## Classes

### MAST
Meta-learning and data Augmentation to Stress Test Forecasting Models **(MAST)** class is used for model evaluation and feature extraction.

Attributes:
- `test_set` (pd.DataFrame): Test dataset for evaluation.
- `model_predictions` (list): List of model predictions (dataframes).
- `models` (list): List of model names.
- `seasonality` (int): Seasonality of the time series data.
- `merged_forecasts` (pd.DataFrame): Merged forecasted predictions.
- `metrics` (list): List of evaluation metrics.
- `large_errors_df` (pd.DataFrame): DataFrame containing large errors.
- `large_errors_ids` (np.ndarray): Array of unique IDs with large errors.
- `features_errors` (pd.DataFrame): DataFrame containing features and errors.

Methods:
- `merge_predictions()`: Merges model predictions with the test set.
- `evaluate_forecasts(train_set)`: Evaluates forecasts using predefined metrics.
- `get_large_errors(quantile, model, metric)`: Identifies large errors.
- `extract_features(train_set, frequency)`: Extracts features from the training set.

### ForecastingModel
Class for training, tuning and forecasting using LightGBM model.

Attributes:
- `frequency` (str): The frequency of the time series data.
- `horizon` (int): Forecast horizon.
- `lags` (list): List of lag values to use for the model.
- `train_set` (pd.DataFrame): Training dataset.
- `lgbm` (AutoMLForecast): LightGBM model.
- `prediction` (pd.DataFrame): Forecasted predictions.
- `execution_time` (float): Time taken for training in minutes.

Methods:
- `train()`: Trains the LightGBM model and performs hyperparameter tuning.
- `forecast()`: Generates forecast using the trained model.
- `create_lag_transforms(lag_transforms, rolling_mean_value)`: Creates lag transforms dictionary.

### BaselineModel
BaselineModel class for training and forecasting using a baseline model.

Attributes:
- `past_df` (pd.DataFrame): Past dataset for training and forecasting.
- `seasonality` (int): Seasonality of the time series data.
- `frequency` (str): Frequency of the time series data.
- `prediction` (pd.DataFrame): Forecasted predictions.
- `execution_time` (float): Time taken for forecasting in minutes.

Methods:
- `forecast(horizon)`: Generates forecast using a baseline model.

### MetaModel
MetaModel class for training a metamodel using LightGBM with optional data resampling.

Attributes:
- `train` (pd.DataFrame): Training dataset.
- `model` (str): Name of the model.
- `resampler` (str): Name of the data resampling technique (default=None).
- `X` (pd.DataFrame): Features.
- `y` (pd.Series): Target variable.
- `resamplers` (dict): Dictionary of resampling techniques.
- `classifier` (LGBMClassifier): The best estimator (classifier) after tuning a lgb model.

Methods:
- `preprocess_set(train)`: Preprocesses the training set, including data resampling if specified.
- `fit_model()`: Fits the LightGBM classifier model, while performing hyperparameter tuning.

### PrepareDataset
Class for loading and preprocessing datasets.

Attributes:
- `directory` (str): Directory path for dataset.
- `dataset` (str): Name of the dataset (M3, M4, Tourism).
- `group` (str): Group name.

### Holdout
Class for hold-out cross-validator generator.

Attributes:
- `n` (int):  Total number of samples.
- `test_size` (float): Fraction of samples to use as test set. Must be a number between 0 and 1.
- `random_state` (int): Seed for the random number generator.
