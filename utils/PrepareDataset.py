import numpy as np
from datasetsforecast.m3 import M3, M3Info
from datasetsforecast.m4 import M4, M4Info
from utils.data.load_data.gluonts_data import GluontsDataset


class PrepareDataset:
    """
    Class to load and prepare datasets for forecasting.

    Attributes:
        dataset (str): Name of the dataset to load. Choose between "M1","M3", or "M4".
        group (str): Name of the dataset group to load.
        directory (str): Directory where the dataset is stored.
        df (pd.DataFrame): DataFrame containing the dataset.
        seasonality (int): Seasonality of the dataset.
        frequency (str): Frequency of the dataset.
        train (pd.DataFrame): DataFrame containing the training set.
        test (pd.DataFrame): DataFrame containing the test set.
        dev_set (pd.DataFrame): DataFrame containing the development set.
        valid (pd.DataFrame): DataFrame containing the validation set.

    Methods:
        load_dataset: Load the dataset.
        train_test_valid_dev_split: Split the dataset into training, test, development and validation sets.
    """

    def __init__(self, dataset, group, directory="utils/data/assets/datasets"):
        self.directory = directory
        self.dataset = dataset
        self.group = group
        self.df = None
        self.seasonality = None
        self.frequency = None
        self.train = None
        self.test = None
        self.dev_set = None
        self.valid = None

    def load_dataset(self):
        match self.dataset:
            # from datasets forecast
            case "M3":
                self.df, *_ = M3.load(directory=self.directory, group=self.group)
                self.seasonality = M3Info[self.group].seasonality
                self.frequency = M3Info[self.group].freq
            case "M4":
                self.df, *_ = M4.load(directory=self.directory, group=self.group)
                self.seasonality = M4Info[self.group].seasonality
                self.frequency = M4Info[self.group].freq
            # from gluonts
            case "M1":
                match self.group:
                    case "Monthly":
                        self.df = GluontsDataset.load_data("m1_monthly")
                        self.seasonality = GluontsDataset.frequency_map["m1_monthly"]
                        self.frequency = GluontsDataset.frequency_pd["m1_monthly"]
                    case "Quarterly":
                        self.df = GluontsDataset.load_data("m1_quarterly")
                        self.seasonality = GluontsDataset.frequency_map["m1_quarterly"]
                        self.frequency = GluontsDataset.frequency_pd["m1_quarterly"]
                    case "Yearly":
                        self.df = GluontsDataset.load_data("m1_yearly")
                        self.seasonality = GluontsDataset.frequency_map["m1_yearly"]
                        self.frequency = GluontsDataset.frequency_pd["m1_yearly"]
                    case _:
                        raise Exception(
                            "Invalid group: either choose Monthly, Quarterly or Yearly"
                        )
            case _:
                raise Exception("Invalid group: either choose M1, M3 or M4")
        # convert "ds" column to int if not a datetime
        if isinstance(self.df["ds"].iloc[0], (int, np.int32, np.int64)):
            self.df["ds"] = self.df["ds"].astype(int)

    def train_test_valid_dev_split(self, horizon):
        self.test = self.df.groupby("unique_id").tail(horizon)
        self.train = self.df.drop(self.test.index).reset_index(drop=True)
        self.valid = self.train.groupby("unique_id").tail(horizon)
        self.dev_set = self.train.drop(self.valid.index).reset_index(drop=True)
