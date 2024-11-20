import numpy as np
from datasetsforecast.m3 import M3, M3Info
from datasetsforecast.m4 import M4, M4Info
from utils.data.codebase.load_data.tourism import TourismDataset


class PrepareDataset:
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
            case "M3":
                self.df, *_ = M3.load(directory=self.directory, group=self.group)
                self.seasonality = M3Info[self.group].seasonality
                self.frequency = M3Info[self.group].freq
            case "M4":
                self.df, *_ = M4.load(directory=self.directory, group=self.group)
                self.seasonality = M4Info[self.group].seasonality
                self.frequency = M4Info[self.group].freq
            case "Tourism":
                self.df = TourismDataset.load_data(self.group)
                self.seasonality = TourismDataset.frequency_map[self.group]
                self.frequency = TourismDataset.frequency_pd[self.group]
            case _:
                raise Exception("Invalid group: either choose M3, M4 or Tourism")
        # convert series id to a categorical feature
        # self.df["unique_id"] = self.df["unique_id"].astype("category")
        # convert "ds" column to int if not a datetime
        if isinstance(self.df["ds"].iloc[0], (int, np.int32, np.int64)):
            self.df["ds"] = self.df["ds"].astype(int)

    def train_test_valid_dev_split(self, horizon):
        self.test = self.df.groupby("unique_id").tail(horizon)
        self.train = self.df.drop(self.test.index).reset_index(drop=True)
        self.valid = self.train.groupby("unique_id").tail(horizon)
        self.dev_set = self.train.drop(self.valid.index).reset_index(drop=True)
