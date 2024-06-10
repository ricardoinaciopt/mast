import pandas as pd
from functools import partial
from utilsforecast.losses import *
from utilsforecast.evaluation import evaluate
from tsfeatures import tsfeatures


class MAST:
    """
    Meta-learning and data Augmentation for Stress Testing (MAST).
    Class for model evaluation and feature extraction.

    Attributes:
        test_set (pd.DataFrame): Test dataset for evaluation.
        model_predictions (list): List of model predictions (dataframes).
        models (list): List of model names.
        seasonality (int): Seasonality of the time series data.
        merged_forecasts (pd.DataFrame): Merged forecasted predictions.
        metrics (list): List of evaluation metrics.
        large_errors_df (pd.DataFrame): DataFrame containing large errors.
        large_errors_ids (np.ndarray): Array of unique IDs with large errors.
        features_errors (pd.DataFrame): DataFrame containing features and errors.

    Methods:
        merge_predictions(): Merges model predictions with the test set.
        evaluate_forecasts(train_set): Evaluates forecasts using predefined metrics.
        get_large_errors(quantile, model, metric): Identifies large errors.
        extract_features(train_set, frequency): Extracts features from the training set.
    """

    def __init__(self, test_set, model_predictions, models, seasonality):
        """
        Initializes the MAST object.

        Args:
            test_set (pd.DataFrame): Test dataset for evaluation.
            model_predictions (list): List of model predictions (dataframes).
            models (list): List of model names.
            seasonality (int): Seasonality of the time series data.
        """
        self.test_set = test_set
        self.model_predictions = model_predictions
        self.merged_forecasts = None
        self.models = models
        self.metrics = [partial(mase, seasonality=seasonality), smape]
        self.large_errors_df = None
        self.large_errors_ids = None
        self.features_errors = None

    def merge_predictions(self):
        """
        Merges model predictions with the test set.
        """
        self.merged_forecasts = self.test_set.copy()

        for prediction_df in self.model_predictions:
            self.merged_forecasts = pd.merge(
                self.merged_forecasts,
                prediction_df,
                on=["unique_id", "ds"],
                how="inner",
            )

    def evaluate_forecasts(self, train_set):
        """
        Evaluates forecasts using predefined metrics.

        Args:
            train_set (pd.DataFrame): Training dataset.
        """
        self.merge_predictions()
        self.evaluation = evaluate(
            self.merged_forecasts,
            metrics=self.metrics,
            models=self.models,
            train_df=train_set,
        )
        self.summary = (
            self.evaluation.drop(columns="unique_id")
            .groupby("metric")
            .mean()
            .reset_index()
        )

    def get_large_errors(self, quantile, model, metric):
        """
        Identifies large errors.

        Args:
            quantile (float): Quantile value for error threshold.
            model (str): Model name.
            metric (str): Evaluation metric.
        """
        self.errors_df = self.evaluation.copy()
        self.errors_df = self.errors_df[self.errors_df["metric"] == metric]

        # calculate the nth percentile of the errors (gross/large errors), for the desired model
        percentile_n = self.errors_df[model].quantile(quantile)
        self.large_errors_df = self.errors_df[
            (self.errors_df[model] > percentile_n)
        ].copy()
        self.large_errors_ids = self.large_errors_df["unique_id"].unique()

    def extract_features(self, train_set, frequency):
        """
        Extracts features from the training set.

        Args:
            train_set (pd.DataFrame): Training dataset.
            frequency (str): Frequency of the time series data.
        """
        self.features_errors = pd.merge(
            self.errors_df.copy(),
            tsfeatures(train_set, freq=frequency),
            on="unique_id",
            how="inner",
        )
        self.features_errors.drop(columns=["SeasonalNaive"], inplace=True)

        self.features_errors["large_error"] = (
            self.features_errors["unique_id"].isin(self.large_errors_ids).astype(int)
        )
