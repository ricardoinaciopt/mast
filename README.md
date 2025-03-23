# Meta-learning and data Augmentation to Stress Test Forecasting Models (MAST)

Repository complementary to the paper `MAST`: Meta-learning and data Augmentation to Stress Test Forecasting Models

## Usage

To use the classes and scripts provided in this repository, follow the instructions outlined in the documentation and code comments. Make sure to install the required dependencies listed in the `requirements.txt` file before running any scripts.

To initiate experiments and evaluate different models and resampling techniques, use `main.py`. Specify the scripts and resamplers to use within `main.py` before execution.

For a faster execution, `main_parallel.py` is available, in which severall settings can be specified for parallel processing across available CPUs.

### Running Experiments

Experiments are divided into three groups: Monthly, Quarterly, and Yearly. Each group has a different number of periods for the horizon. The following commands can be used to run each type of experiment:

- For large errors experiment:
  ```sh
  python main.py --experiment large_errors
  ```

- For large uncertainty experiment:
  ```sh
  python main.py --experiment large_uncertainty
  ```

- For large error but low uncertainty (large certainty) experiment:
  ```sh
  python main.py --experiment large_certainty
  ```

The `main.py` script will generate the pipelines for the experiments based on the specified experiment type and the dataset groups (Monthly, Quarterly, Yearly).

Alternativly, one can also specify the correct experiment in `main.py`, by updating the script path in the `command` list:

```python
command = [
    "python",
    # ./errors_uid_pipeline.py for large errors experiment
    # ./uncertainty_uid_pipeline.py for uncertainty experiment
    # ./hubris_uid_pipeline.py for hubris (large error but large certainty) experiment
    "./errors_uid_pipeline.py",  # Change this to the appropriate script
    "--data",
    data,
    "--group",
    group,
    "--horizon",
    h,
    "--models",
    "LGBM",
]
```

## Classes Overview

This repository includes several Python classes designed for different stages of the metamodel creation for predicting and evaluating stress testing and conditions:

- `MAST`: Class for model evaluation and feature extraction.
- `ForecastingModel`: Class for training, tuning and forecasting using a LightGBM model.
- `BaselineModel`: Class for training and forecasting using a baseline model.
- `MetaModel`: Class for training a metamodel using LightGBM with optional data resampling.
- `PrepareDataset`: Class for loading and preprocessing datasets.
- `TimeSeriesGenerator`: Class for generating synthetic time series data using the MetaForecast package.

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
Class for training and forecasting using a specified model or list of models with hyperparameter tuning.
Model tuning is done using AutoModel from Nixtla and optuna.
It takes sklearn-compatible regressors, and wraps them around an AutoModel object to find the best hyperparameters and make forecasts. if necessary, an interval of confidence can be specified.

Attributes:
- `models` (str or list): Model(s) to prepare for evaluation.
- `train_set` (pd.DataFrame): Training dataset.
- `frequency` (str): The frequency of the time series data.
- `horizon` (str): The forecasting horizon.
- `seasonality` (str): The seasonal length of the time series data.

Methods:
- `train()`: Trains the forecasting oject, on the specified model(s) and performs hyperparameter tuning.
- `forecast()`: Generates forecast using the forecasting AutoMLForecast object.

### BaselineModel
BaselineModel class for training and forecasting using a baseline model.

Attributes:
- `past_df` (pd.DataFrame): Past dataset for training and forecasting.
- `seasonality` (int): Seasonality of the time series data.
- `frequency` (str): Frequency of the time series data.
- `prediction` (pd.DataFrame): Forecasted predictions.


### MetaModel
MetaModel class for training a metamodel using LightGBM with optional data resampling.

Attributes:
- `train` (pd.DataFrame): Training dataset.
- `model` (str): Name of the model.
- `resampler` (str): Name of the data resampling technique (default=None).
- `X` (pd.DataFrame): Features.
- `y` (pd.Series): Target variable.
- `resamplers` (dict): Dictionary of resampling techniques.
- `classifier` (LGBMClassifier or XGBoostClassifier): The best estimator (classifier) after tuning a given model.

Methods:
- `preprocess_set(train)`: Preprocesses the training set, including data resampling if specified.
- `fit_model()`: Fits the classifier model, while performing hyperparameter tuning, if specified.

### PrepareDataset
Class for loading and preprocessing datasets.

Attributes:
- `directory` (str): Directory path for dataset.
- `dataset` (str): Name of the dataset (M3, M4, Tourism).
- `group` (str): Group name.

### TimeSeriesGenerator

TimeSeriesGenerator class for generating synthetic time series data using the MetaForecast package. This class serves as a wrapper around various augmentation methods to provide a unified interface for generating enriched time series datasets.

Attributes:
- `df` (pd.DataFrame): Input DataFrame containing time series data. Assumes a `unique_id` column for identifying time series and optionally a target column for filtering.
- `seasonality` (int or None): Specifies the seasonal period of the time series (e.g., 12 for monthly data).
- `frequency` (str or None): Frequency of the time series (e.g., "M" for monthly, "Q" for quarterly).
- `min_len` (int or None): Minimum length of the time series for augmentation.
- `max_len` (int or None): Maximum length of the time series for augmentation.
- `methods` (dict): A dictionary mapping method names to their respective classes and pre-configured instances of MetaForecast augmenters.
- `target` (str): Specifies the target variable to use for synthetic generation. Can be one of "errors" (large errors), "uncertainty" (large uncertainty), or "certainty" (low errors and uncertainty).

Methods:
- `get_class_methods(cls)`: Returns a list of method names defined in the specified class.
- `generate_synthetic_dataset(method_name, n_samples=100)`: Generates synthetic datasets using the specified augmentation method.



