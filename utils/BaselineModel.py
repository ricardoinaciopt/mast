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

    Methods:
        forecast(horizon): Generates forecast using a baseline model.
    """

    def __init__(self, past_df, seasonality, frequency, horizon):
        """
        Initializes the BaselineModel object.

        Args:
            past_df (pd.DataFrame): Past dataset for training and forecasting.
            seasonality (int): Seasonality of the time series data.
            frequency (str): Frequency of the time series data.
            horizon (int): Forecast horizon.
        """
        self.past_df = past_df
        self.frequency = frequency
        self.seasonality = seasonality
        self.horizon = horizon
        self.prediction = None
        self.baseline = None
        self.baseline = StatsForecast(
            models=[SeasonalNaive(season_length=self.seasonality)],
            freq=self.frequency,
            n_jobs=-1,
        )
        self.prediction = self.baseline.forecast(df=self.past_df, h=self.horizon)
