from time import time
from mlforecast.target_transforms import Differences
from window_ops.rolling import rolling_mean
from mlforecast.auto import AutoModel, AutoMLForecast
import lightgbm as lgb
import optuna


class ForecastingModel:
    """
    Class for training and forecasting using LightGBM model hyperparameter tuning.
    Model is lightgbm, and tuning is done using AutoModel from Nixtla and optuna.

    Attributes:
        frequency (str): The frequency of the time series data.
        lags (list): List of lag values to use for the model.
        train_set (pd.DataFrame): Training dataset.
        lgbm (AutoMLForecast): LightGBM model.
        prediction (pd.DataFrame): Forecasted predictions.
        execution_time (float): Time taken for training in minutes.
    """

    def __init__(
        self,
        frequency,
        horizon,
        lags,
        train,
    ):
        """
        Initializes the ForecastingModel object.

        Args:
            frequency (str): The frequency of the time series data.
            target_transforms (str): Type of target transform to apply.
            horizon (int): Forecast horizon.
            lags (list): List of lag values to use for the model.
            lag_transforms (str): Type of lag transform to apply.
            rolling_mean (function): Rolling mean function value.
            train (pd.DataFrame): Training dataset.
        """
        self.frequency = frequency
        self.horizon = horizon
        self.lags = lags
        self.train_set = train
        self.lgbm = None
        self.prediction = None
        self.execution_time = None

    def train(self):
        """
        Trains the LightGBM model using cross-validation.
        """

        # fixed parameters
        lgbm_params = {
            "random_seed": 42,
            "boosting_type": "gbdt",
            "force_col_wise": True,
            "verbosity": -1,
        }

        # tunable parameters
        def my_lgb_config(trial: optuna.Trial):
            return {
                "learning_rate": trial.suggest_categorical(
                    "learning_rate", [0.01, 0.02, 0.05]
                ),
                "num_leaves": trial.suggest_categorical("num_leaves", [2, 32, 64, 128]),
                "max_depth": trial.suggest_categorical("max_depth", [5, 10, 15]),
                "n_estimators": trial.suggest_categorical(
                    "n_estimators", [50, 100, 200]
                ),
            }

        def init_config(trial: optuna.Trial):
            return {
                "lags": [i for i in range(1, 12)],
                "lag_transforms": self.create_lag_transforms([12], 12),
                "target_transforms": [Differences([12])],
            }

        tuned_lgb = AutoModel(
            model=lgb.LGBMRegressor(**lgbm_params), config=my_lgb_config
        )

        init = time()
        self.lgbm = AutoMLForecast(
            models={"ForecastingModel": tuned_lgb},
            freq=self.frequency,
            season_length=12,
            init_config=init_config,
        ).fit(self.train_set, h=self.horizon, num_samples=10, n_windows=2)
        end = time()
        self.execution_time = (end - init) / 60  # Time in minutes

    def forecast(self):
        """
        Generates forecast using the trained model.
        """
        self.prediction = self.lgbm.predict(self.horizon)
        self.prediction.rename(
            columns={"LGBMRegressor": "ForecastingModel"}, inplace=True
        )

    def create_lag_transforms(self, lag_transforms, rolling_mean_value):
        """
        Create lag transforms dictionary.

        Args:
            lag_transforms (list): List of lag values.

        Returns:
            dict: Lag transforms dictionary.
        """
        result = {}
        for lag in lag_transforms:
            result[lag] = [(rolling_mean, rolling_mean_value)]
        return result
