from metaforecast.synth.generators.tsmixup import TSMixup
from metaforecast.synth.generators.kernelsynth import KernelSynth
from metaforecast.synth.generators.mbb import SeasonalMBB
from metaforecast.synth.generators.dba import DBA
from metaforecast.synth.generators.jittering import Jittering
from metaforecast.synth.generators.scaling import Scaling
from metaforecast.synth.generators.warping_mag import MagnitudeWarping
from metaforecast.synth.generators.warping_time import TimeWarping

import pandas as pd
import inspect


class TimeSeriesGenerator:
    """
    Unified wrapper for synthetic time series generation methods
    from the MetaForecast package. It enables seamless augmentation of time series
    datasets using various techniques such as mixing, kernel synthesis, magnitude
    warping, jittering, and more.

    Attributes:
    - df (pd.DataFrame):
        Input DataFrame containing the time series data.
    - seasonality (int or None):
        Specifies the seasonal period of the time series. Required for seasonal-specific
        methods.
    - frequency (str or None):
        Indicates the frequency of the time series.
    - min_len (int or None):
        Minimum length of time series to consider during augmentation.
    - max_len (int or None):
        Maximum length of time series to consider during augmentation.
    - methods (dict):
        Maps augmentation method names to their respective MetaForecast class and
        initialized objects, pre-configured for the provided attributes.

    Methods:
        get_class_methods(type):
        Returns the list of method names defined within a given class.

        generate_synthetic_dataset(method_name, n_samples = 100):
        Generates a synthetic dataset using the specified augmentation method. Supports
        multiple augmentation techniques, both `transform`-based and custom synthetic generation.
    """

    def __init__(
        self, df, seasonality=None, frequency=None, min_len=None, max_len=None
    ):
        """
        Initializes the TimeSeriesGenerator with dataset attributes and pre-configures
        augmentation methods.

        Args:
            df (pd.DataFrame): The input dataset containing time series data.
            seasonality (int or None): Seasonal period of the data.
            frequency (str or None): Frequency of the data.
            min_len (int or None): Minimum time series length for augmentation.
            max_len (int or None): Maximum time series length for augmentation.
        """
        self.df = df
        self.seasonality = seasonality
        self.frequency = frequency
        self.min_len = min_len
        self.max_len = max_len

        self.methods = {
            "TSMixup": [
                TSMixup,
                TSMixup(max_n_uids=3, min_len=self.min_len, max_len=self.min_len),
            ],
            "KernelSynth": [
                KernelSynth,
                KernelSynth(max_kernels=5, freq=self.frequency, n_obs=self.min_len),
            ],
            "DBA": [DBA, DBA(max_n_uids=3)],
            "Scaling": [Scaling, Scaling()],
            "MagnitudeWarping": [MagnitudeWarping, MagnitudeWarping()],
            "TimeWarping": [TimeWarping, TimeWarping()],
            "SeasonalMBB": [SeasonalMBB, SeasonalMBB(seas_period=self.seasonality)],
            "Jittering": [Jittering, Jittering()],
        }

    def get_class_methods(self, cls):
        """
        Returns the names of all methods defined within a given class.

        Args:
            cls (type): The class to inspect.
        """
        methods = inspect.getmembers(cls, predicate=inspect.isfunction)

        class_methods = [
            name
            for name, func in methods
            if func.__qualname__.startswith(cls.__name__ + ".")
        ]
        return class_methods

    def generate_synthetic_dataset(self, method_name, n_samples=100):
        """
        Generates a synthetic dataset using the specified augmentation method.

        Args:
            method_name (str): Name of the augmentation method to use.
            n_samples (int, default=100): Number of synthetic samples to generate.
        """
        if method_name in {"DBA", "TSMixup"}:
            self.df["unique_id"] = self.df["unique_id"].astype("str")
        method = self.methods.get(method_name)[1]
        cls = self.methods.get(method_name)[0]
        if not method:
            raise ValueError(f"Unknown method_name: {method_name}")
        df = self.df[self.df["large_error"] == 1].drop(columns="large_error")
        augmented_dfs = []

        if "transform" in self.get_class_methods(cls):
            n_samples *= 100
            if method_name == "KernelSynth":
                augmented_df = method.transform(n_samples)
            else:
                augmented_df = method.transform(df, n_samples)
            augmented_df["unique_id"] = augmented_df["unique_id"].astype(str) + "_SYN"
            augmented_dfs.append(augmented_df.copy())
        else:
            for i in range(n_samples):
                augmented_df = method._create_synthetic_ts(df)
                augmented_df["unique_id"] = (
                    augmented_df["unique_id"].astype(str).str.split("_").str[0]
                    + f"_SYN{i+1}"
                )
                augmented_dfs.append(augmented_df.copy())

        return pd.concat(augmented_dfs, axis=0).reset_index(drop=True)