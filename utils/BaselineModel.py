from time import time
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive


class BaselineModel:
    """
    BaselineModel class for training and forecasting using a baseline model.

    Attributes:
        past_df (pd.DataFrame): Past dataset for training and forecasting.
        seasonality (int): Seasonality of the time series data.
        frequency (str): Frequency of the time series data.
        prediction (pd.DataFrame): Forecasted predictions.
        execution_time (float): Time taken for forecasting in minutes.

    Methods:
        forecast(horizon): Generates forecast using a baseline model.
    """

    def __init__(self, past_df, seasonality, frequency):
        """
        Initializes the BaselineModel object.

        Args:
            past_df (pd.DataFrame): Past dataset for training and forecasting.
            seasonality (int): Seasonality of the time series data.
            frequency (str): Frequency of the time series data.
        """
        self.past_df = past_df
        self.frequency = frequency
        self.seasonality = seasonality
        self.prediction = None
        self.execution_time = None

    def forecast(self, horizon):
        """
        Generates forecast using a baseline model.

        Args:
            horizon (int): Forecast horizon.
        """
        sf = StatsForecast(
            models=[SeasonalNaive(season_length=self.seasonality)],
            freq=self.frequency,
            n_jobs=-1,
        )
        init = time()
        self.prediction = sf.forecast(df=self.past_df, h=horizon)
        end = time()
        self.execution_time = (end - init) / 60  # Time in minutes
