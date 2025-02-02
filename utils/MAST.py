import os
import pandas as pd
from scipy.stats import rankdata
from functools import partial
from utilsforecast.losses import *
from utilsforecast.evaluation import evaluate
from tsfeatures import tsfeatures
from sklearn.preprocessing import StandardScaler


class MAST:
    """
    Meta-learning and data Augmentation for Stress Testing (MAST).
    Class for model evaluation and feature extraction.
    Computes which timeseries put the forecasting model under stress.
    Identifies instances (timeseries) of large errors and large uncertainty.

    Attributes:
        test_set (pd.DataFrame): Test dataset for evaluation.
        model_predictions (list): List of model predictions (dataframes).
        models (list): List of model names.
        seasonality (int): Seasonality of the time series data.
        merged_forecasts (pd.DataFrame): Merged forecasted predictions.
        metrics (list): List of evaluation metrics.
        large_errors_df (pd.DataFrame): DataFrame containing large errors.
        large_uncertainty_df (pd.DataFrame): DataFrame containing large uncertainty.
        large_errors_ids (np.ndarray): Array of unique IDs with large errors.
        large_uncertainty_ids (np.ndarray): Array of unique IDs with large uncertainty.
        features_errors (pd.DataFrame): DataFrame containing features and errors.
        avg_uncertainty_df (pd.DataFrame): DataFrame containing the avergae interval size for each timeseries.
        interval_summary (pd.DataFrame): DataFrame containing the average interval size for each model.
        error_summary (pd.DataFrame): DataFrame containing the average error for each model.

    Methods:
        merge_predictions(): Merges model predictions with the test set.
        evaluate_forecasts(train_set): Evaluates forecasts using predefined metrics.
        get_large_errors(quantile, model, metric): Identifies large errors.
        extract_features(train_set, frequency): Extracts features from the training set.
    """

    def __init__(self, test_set, model_predictions, metrics, seasonality):
        """
        Initializes the MAST object.

        Args:
            test_set (pd.DataFrame): Test dataset for evaluation.
            model_predictions (list): List of model predictions (dataframes).
            metrics (list): List of metric names.
            seasonality (int): Seasonality of the time series data.
        """
        self.test_set = test_set
        self.model_predictions = model_predictions
        self.merged_forecasts = None
        self.seasonality = seasonality
        self.metrics = self._initialize_metrics(metrics)
        self.large_errors_df = None
        self.large_errors_ids = None
        self.interval_summary = None
        self.avg_uncertainty_df = None
        self.large_uncertainty_df = None
        self.large_uncertainty_ids = None
        self.features_errors = None

    def _initialize_metrics(self, metrics):
        """
        Defines the metrics to compute the prediction errors.

        Args:
            metrics (list): List of metrics to compute errors.
        """
        metric_dict = {
            "mase": partial(mase, seasonality=self.seasonality),
            "smape": smape,
        }
        return [metric_dict[metric] for metric in metrics]

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
            train_df=train_set,
        )
        self.error_summary = (
            self.evaluation.drop(columns="unique_id")
            .groupby("metric")
            .mean()
            .reset_index()
        )

    def get_large_errors(self, model, metric, quantile=None):
        """
        Identifies large errors.

        Args:
            quantile (float): Quantile value for error threshold.
            model (str): Model name.
            metric (str): Evaluation metric.
        """
        self.errors_df = self.evaluation.copy()
        self.errors_df = self.errors_df[self.errors_df["metric"] == metric]

        # compute error quantile
        self.errors_df[f"error_quantile_{model}"] = (
            (
                rankdata(self.errors_df[model], method="average")
                / len(self.errors_df)
                * 100
            )
        ).astype(int)

        if quantile != None:
            print(f"Large Errors threshold setting: {int(quantile*100)}th quantile.")
            # calculate the nth percentile of the errors (large errors), for the desired model
            percentile_n = self.errors_df[model].quantile(quantile)
            self.large_errors_df = self.errors_df[
                (self.errors_df[model] > percentile_n)
            ]
        else:
            print("Large Errors threshold setting: 2*std.")
            # compute mean and std, and get those that are N times away from the std
            lgbm_mean = self.errors_df[model].mean()
            lgbm_std = self.errors_df[model].std()
            large_std = self.errors_df[model] > lgbm_mean + (2 * lgbm_std)

            self.large_errors_df = self.errors_df[large_std]

        self.large_errors_ids = self.large_errors_df["unique_id"].unique()

    def compute_uncertainty(self, train_set, level, predictions=None):
        """
        Computes the average of the uncertainty intervals for predictions.

        Args:
            predictions (pd.DataFrame): DataFrame containing the predictions.
            level (int): The confidence level to replace in the column names.
            train_set (pd.DataFrame): Train set DataFrame containing the 'unique_id' and 'y' columns.
        """
        if predictions is None or predictions.empty:
            if not self.merged_forecasts.empty:
                predictions = self.merged_forecasts
            else:
                raise ValueError(
                    'No forecasts available, please use the "evaluate_forecasts()" method first.'
                )
        model_columns = predictions.columns[3:-1]
        models = set(col.split("-")[0] for col in model_columns if "-" in col)
        avg_uncertainty_df = pd.DataFrame()

        for model in models:
            concat_list = []

            lo_col = f"{model}-lo-{level}"
            hi_col = f"{model}-hi-{level}"
            lo_col_scaled = f"{model}-lo-{level}-scaled"
            hi_col_scaled = f"{model}-hi-{level}-scaled"

            for id in predictions["unique_id"].unique():
                scaler = StandardScaler()
                scaler.fit(train_set[train_set["unique_id"] == id][["y"]].values)

                scaled_intervals = predictions[predictions["unique_id"] == id].copy()
                scaled_intervals[lo_col_scaled] = scaler.transform(
                    scaled_intervals[[lo_col]]
                )
                scaled_intervals[hi_col_scaled] = scaler.transform(
                    scaled_intervals[[hi_col]]
                )

                scaled_intervals["scaled_interval_size"] = (
                    scaled_intervals[hi_col_scaled] - scaled_intervals[lo_col_scaled]
                )
                concat_list.append(
                    scaled_intervals[
                        [
                            "unique_id",
                            "ds",
                            lo_col_scaled,
                            hi_col_scaled,
                            "scaled_interval_size",
                        ]
                    ]
                )

            concat_df = pd.concat(concat_list)
            model_avg_uncertainty = (
                concat_df.groupby("unique_id")["scaled_interval_size"]
                .mean()
                .rename(f"avg_interval_size_{model}")
            )

            avg_uncertainty_df = pd.concat(
                [avg_uncertainty_df, model_avg_uncertainty], axis=1
            )

        self.avg_uncertainty_df = avg_uncertainty_df.reset_index()
        self.avg_uncertainty_df.rename(columns={"index": "unique_id"}, inplace=True)

        for model in models:
            # compute uncertainty quantile
            self.avg_uncertainty_df[f"uncertainty_quantile_{model}"] = (
                rankdata(
                    self.avg_uncertainty_df[f"avg_interval_size_{model}"],
                    method="average",
                )
                / len(self.avg_uncertainty_df)
                * 100
            ).astype(int)

        self.interval_summary = pd.DataFrame()
        for col in self.avg_uncertainty_df.columns[1:]:
            avg_size = self.avg_uncertainty_df[col].mean()
            model_name = col.replace("avg_interval_size_", "interval_summary_")
            self.interval_summary[model_name] = [round(avg_size, 3)]

        return self.avg_uncertainty_df

    def get_large_uncertainty(self, model, quantile):
        """
        Identifies predictions with large uncertainty intervals for a specified model.

        Args:
            model (str): The name of the model to analyze (e.g., 'DecisionTree', 'Ridge').
            quantile (float): Quantile value for uncertainty threshold.
        """
        col = f"avg_interval_size_{model}"

        if col not in self.avg_uncertainty_df.columns:
            raise ValueError(f"{model} not found.")

        if quantile != None:
            print(
                f"Large Uncertainty threshold setting: {int(quantile*100)}th quantile."
            )

            percentile_n_i = self.avg_uncertainty_df[col].quantile(quantile)
            self.large_uncertainty_df = self.avg_uncertainty_df[
                (self.avg_uncertainty_df[col] > percentile_n_i)
            ]
            self.large_uncertainty_ids = self.large_uncertainty_df["unique_id"].unique()
            return self.large_uncertainty_ids
        raise ValueError("'quantile' must be defined.")

    def get_large_certainty(self, model, quantile):
        """
        Identifies predictions with large certainty intervals for a specified model.

        Args:
            model (str): The name of the model to analyze.
            quantile (float): Quantile value for certainty threshold.
        """
        col = f"avg_interval_size_{model}"

        if col not in self.avg_uncertainty_df.columns:
            raise ValueError(f"{model} not found.")

        if quantile != None:
            print(f"Large Certainty threshold setting: {int(quantile*100)}th quantile.")

            percentile_n_i = self.avg_uncertainty_df[col].quantile(quantile)
            self.large_certainty_df = self.avg_uncertainty_df[
                (self.avg_uncertainty_df[col] <= percentile_n_i)
            ]
            self.large_certainty_ids = self.large_certainty_df["unique_id"].unique()

            return self.large_certainty_ids
        raise ValueError("'quantile' must be defined.")

    def extract_features(self, train_set, filename=None):
        """
        Extracts features from the training set, and saves it to a file. If a filename is specified, that metadataset is used instead.

        Args:
            train_set (pd.DataFrame): Training dataset.
            filename (str): File containing the already extracted features, to reduce compute times.
        """
        if filename:
            try:
                datapath = os.path.join("metadatasets", filename)
                features = pd.read_csv(datapath)
            except FileNotFoundError:
                print(f"The metadataset '{filename}' does not exist. Creating it...")
                features = tsfeatures(train_set, freq=self.seasonality)
                if not os.path.exists(datapath):
                    os.makedirs("metadatasets", exist_ok=True)
                    features.to_csv(datapath, index=False)
                    print(f"Metadataset saved on: '{datapath}'.")
        else:
            features = tsfeatures(train_set, freq=self.seasonality)
            datapath = os.path.join("metadatasets", filename)
            if not os.path.exists(datapath):
                os.makedirs("metadatasets", exist_ok=True)
                features.to_csv(datapath, index=False)
                print(f"Metadataset saved on: '{datapath}'.")

        self.features_errors = pd.merge(
            self.errors_df,
            features,
            on="unique_id",
            how="inner",
        )
        self.features_errors.drop(columns=["SeasonalNaive"], inplace=True)
        self.features_errors.fillna(0, inplace=True)
        self.features_errors["large_error"] = (
            self.features_errors["unique_id"].isin(self.large_errors_ids).astype(int)
        )

    def select_best(self):
        """
        Identifies the best algorithm for the given task, regarding errors and / or uncertainty.
        """
        average_errors = self.error_summary.iloc[:, 1:].mean()
        smallest_avg_error = average_errors.idxmin()
        print("\nBest ForecastingModel: ", smallest_avg_error)
        if self.interval_summary:
            average_uncertainty = self.interval_summary.iloc[:, 1:].mean()
            smallest_avg_uncertainty = average_uncertainty.idxmin()
            return smallest_avg_error, smallest_avg_uncertainty
        return smallest_avg_error
