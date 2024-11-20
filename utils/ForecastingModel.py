import re
from catboost import CatBoostRegressor
from mlforecast.utils import PredictionIntervals
from mlforecast.target_transforms import Differences

from mlforecast.auto import (
    AutoModel,
    AutoMLForecast,
    AutoLightGBM,
    AutoXGBoost,
    AutoLinearRegression,
    AutoRidge,
    AutoLasso,
    AutoElasticNet,
    AutoRandomForest,
    ridge_space,
    lasso_space,
    catboost_space,
    linear_regression_space,
    elastic_net_space,
    random_forest_space,
)
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
import optuna


class ForecastingModel:
    """
    Class for training and forecasting using a specified model or list of models with hyperparameter tuning.
    Model tuning is done using AutoModel from Nixtla and optuna.
    It takes sklearn-compatible regressors, and wraps them around an AutoModel object to find the best hyperparameters and make forecasts. if necessary, an interval of confidence can be specified.

    Attributes:
        models (str or list): Model(s) to prepare for evaluation.
        train_set (pd.DataFrame): Training dataset.
        frequency (str): The frequency of the time series data.
        horizon (str): The forecasting horizon.
        seasonality (str): The seasonal length of the time series data.
    """

    def __init__(
        self,
        models,
        train_set,
        frequency,
        horizon,
        seasonality,
    ):
        """
        Initializes the ForecastingModel object.

        Args:
            models (str or list): Model(s) to prepare for evaluation.
            regressor (AutoMLForecast): The regressor object, that contains the model, or the list of tuned models.
            train_set (pd.DataFrame): Training dataset.
            frequency (str): The frequency of the time series data.
            horizon (str): The forecasting horizon.
            seasonality (str): The seasonal length of the time series data.
            prediction (Dataframe): Result of the predictions made by the regressor object.
        """
        self.frequency = frequency
        self.horizon = horizon
        self.seasonality = seasonality
        self.train_set = train_set
        self.models = None
        self.regressor = None
        self.prediction = None

        # handle models that don't support categorical features, enconding them
        def handle_categoric(model, model_name):
            cat_pipeline = make_pipeline(
                ColumnTransformer(
                    [
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                            ["unique_id"],
                        ),
                    ],
                ),
                model(),
            )
            space = {
                "ridge": ridge_space,
                "lasso": lasso_space,
                "linearregression": linear_regression_space,
                "elasticnet": elastic_net_space,
                "randomforestregressor": random_forest_space,
                "catboostregressor": catboost_space,
            }.get(model_name)
            return AutoModel(
                cat_pipeline,
                lambda trial: {
                    f"{model_name}__{k}": v for k, v in space(trial).items()
                },
            )

        # supported models / algorithms
        self.supported_models = {
            "LGBM": AutoLightGBM(),
            "ElasticNet": handle_categoric(ElasticNet, "elasticnet"),
            "XGB": AutoXGBoost(
                config=lambda trial: {"enable_categorical": True},
            ),
            "Lasso": handle_categoric(Lasso, "lasso"),
            "Ridge": handle_categoric(Ridge, "ridge"),
            "LinearRegression": handle_categoric(LinearRegression, "linearregression"),
            "CatBoost": handle_categoric(CatBoostRegressor, "catboostregressor"),
        }
        self._init_model(models)

    def _init_model(self, models):
        """
        Initializes the model(s), including optimization search space(s).
        Prepares the regressor(s) for training, and calls the train() method.
        """
        if isinstance(models, str):
            models = [models]
        self.models = {
            model: self.supported_models[model]
            for model in models
            if model in self.supported_models
        }
        self.train()

    def train(self):
        """
        Trains the forecasting model and performs hyperparameter tuning.
        """

        def init_config(trial: optuna.Trial):
            """
            12th difference for monthly data.
            4th difference for quarterly data.
            1st difference for yearly data.
            Account for frequencies at season start and end.
            """
            difference = {
                "M": 12,
                "MS": 12,
                "Q": 4,
                "QS": 4,
            }.get(self.frequency, 1)
            return {
                "lags": [i for i in range(1, self.horizon)],
                "target_transforms": [Differences([difference])],
            }

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        self.regressor = AutoMLForecast(
            models=self.models,
            freq=self.frequency,
            season_length=self.seasonality,
            init_config=init_config,
            # fit_config=lambda trial: {"static_features": ["unique_id"]},
            num_threads=len(self.models),
        ).fit(
            df=self.train_set,
            h=self.horizon,
            num_samples=10,
            n_windows=2,
            # prediction_intervals=PredictionIntervals(h=self.horizon),
        )

    def forecast(self, level):
        """
        Generates forecast using the trained AutoMLForecast object.
        """
        self.prediction = self._simplify_names(
            self.regressor.predict(h=self.horizon, level=[level])
        )

    def _simplify_names(self, df):
        """
        Simplifies the column names of a pandas Dataframe, for improved readability.
        """
        df.columns = [
            re.compile(r"\([^)]*\)").sub("", str(col)).replace("Regressor", "").strip()
            for col in df.columns
        ]
        return df
