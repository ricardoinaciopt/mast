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
        df (pd.DataFrame): Input DataFrame containing the time series data.
        dataset (str or None): Specifies the selected dataset, to augment with a given method.
        group (str or None): Specifies the sampling frequency of the dataset used.
        seasonality (int or None): Specifies the seasonal period of the time series. Required for seasonal-specific methods.
        frequency (str or None): Indicates the frequency of the time series.
        min_len (int or None): Minimum length of time series to consider during augmentation.
        max_len (int or None): Maximum length of time series to consider during augmentation.
        methods (dict): Maps augmentation method names to their respective MetaForecast class and initialized objects pre-configured for the provided attributes.
        target (str): Specifies the target variable to use for synthetic generation. Can be one of "errors" (large errors), "uncertainty" (large uncertainty), or "certainty" (low errors and uncertainty).

    Methods:
        get_class_methods(cls): Returns the names of all methods defined within a given class.
        generate_synthetic_dataset(method_name): Generates a synthetic dataset using the specified augmentation method and returns the augmented DataFrame.
    """

    def __init__(
        self,
        df,
        dataset=None,
        group=None,
        seasonality=None,
        frequency=None,
        min_len=None,
        max_len=None,
        target=None,
    ):
        """
        Initializes the TimeSeriesGenerator with dataset attributes and pre-configures
        augmentation methods.

        Args:
            df (pd.DataFrame): The input dataset containing time series data.
            dataset (str or None): The selected dataset, to augment with a given method.
            group (str or None): The sampling frequency of the dataset used.
            seasonality (int or None): Seasonal period of the data.
            frequency (str or None): Frequency of the data.
            min_len (int or None): Minimum time series length for augmentation.
            max_len (int or None): Maximum time series length for augmentation.
            target (str or None): Target variable for synthetic generation.
        """
        self.df = df
        self.dataset = dataset
        self.group = group
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
        if target == "errors":
            self.target = "large_error"
        elif target == "uncertainty":
            self.target = "large_uncertainty"
        elif target == "certainty":
            self.target = "le_lc"
        else:
            raise ValueError('Invalid target parameter. Use "errors" or "uncertainty".')

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

    def generate_synthetic_dataset(self, method_name):
        """
        Generates a synthetic dataset using the specified augmentation method.

        Args:
            method_name (str): Name of the augmentation method to use.
        """
        gen_diff = self.df[self.target].value_counts().get(0, 0) - self.df[
            self.target
        ].value_counts().get(1, 0)

        # decrease the number of series to generate, to reduce computation times
        # these values are arbitrary and can be adjusted, as each was chosen empirically

        # size of generation based on dataset, group, and approach (transform or loop)
        gen_size_transform1, gen_size_transform2, gen_size_loop = {
            ("M4", "Monthly"): (200, 200, 1000000),
            ("M4", "Quarterly"): (40, 60, 250000),
            ("M4", "Yearly"): (20, 20, 50000),
        }.get((self.dataset, self.group), (40, 60, 10000))
        # factor to reduce based on the sampling frequency
        gen_factor = {
            "M": 1,
            "MS": 1,
            "Q": 4,
            "QS": 4,
            "Y": 8,
            "YS": 8,
        }.get(self.frequency, 1)

        if method_name in {"DBA", "TSMixup"}:
            self.df["unique_id"] = self.df["unique_id"].astype("str")
        method = self.methods.get(method_name)[1]
        cls = self.methods.get(method_name)[0]
        if not method:
            raise ValueError(f"Unknown method_name: {method_name}")
        df = self.df[self.df[self.target] == 1].drop(columns=self.target)
        augmented_dfs = []
        if "transform" in self.get_class_methods(cls):
            if method_name in {"TSMixup", "KernelSynth"}:
                gen_diff = round(gen_diff / (gen_size_transform1 / gen_factor))
            else:
                gen_diff = round(gen_diff / (gen_size_transform2 / gen_factor))
            print("Generating ", gen_diff, " series with: ", method_name)
            if method_name == "KernelSynth":
                augmented_df = method.transform(gen_diff)
            else:
                augmented_df = method.transform(df, gen_diff)
            augmented_df["unique_id"] = augmented_df["unique_id"].astype(str) + "_SYN"
            augmented_dfs.append(augmented_df.copy())
        else:
            gen_diff = round(gen_diff / (gen_size_loop / gen_factor))
            if self.target == "le_lc":
                gen_diff *= 10
            print("Generating ", gen_diff, " series with: ", method_name)

            for i in range(gen_diff):
                augmented_df = method._create_synthetic_ts(df)
                augmented_df["unique_id"] = (
                    augmented_df["unique_id"].astype(str).str.split("_").str[0]
                    + f"_SYN{i+1}"
                )
                augmented_dfs.append(augmented_df.copy())
        print("Done generating.")
        return pd.concat(augmented_dfs, axis=0).reset_index(drop=True)
