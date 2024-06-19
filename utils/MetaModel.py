import lightgbm as lgb
from utils.Holdout import Holdout
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE, ADASYN


class MetaModel:
    """
    MetaModel class for training a meta-model using LightGBM with optional data resampling.

    Attributes:
        train (pd.DataFrame): Training dataset.
        model (str): Name of the model.
        resampler (str): Name of the data resampling technique (default=None).
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
        resamplers (dict): Dictionary of resampling techniques.
        classifier (LGBMClassifier): The best estimator (classifier) after tuning a lgb model.

    Methods:
        preprocess_set(train): Preprocesses the training set, including data resampling if specified.
        fit_model(): Fits the LightGBM classifier model.
    """

    def __init__(self, train_set, model, resampler=None):
        """
        Initializes the MetaModel object.

        Args:
            train_set (pd.DataFrame): Training dataset.
            model (str): Name of the model.
            resampler (str): Name of the data resampling technique (default=None).
        """
        self.model = model
        self.resampler = resampler
        self.train = train_set.copy()
        self.X = None
        self.y = None
        self.resamplers = {
            "SMOTE": SMOTE(),
            "ADASYN": ADASYN(),
        }
        self.classifier = None
        self.preprocess_set(self.train)

    def preprocess_set(self, train):
        """
        Preprocesses the training set, including data resampling if specified.

        Args:
            train (pd.DataFrame): Training dataset.
        """
        train.set_index("unique_id", inplace=True)
        train.drop(columns=["metric"], inplace=True)
        train.fillna(0, inplace=True)

        if self.resampler and self.resampler in self.resamplers:
            data_to_resample = self.train.copy()
            large_errors_mask = self.train["large_error"].astype(int)

            resampled_data, err_class = self.resamplers[self.resampler].fit_resample(
                data_to_resample, large_errors_mask
            )
            self.X = resampled_data.drop(["large_error", str(self.model)], axis=1)
            self.y = resampled_data["large_error"]
        else:
            self.X = train.drop(["large_error", str(self.model)], axis=1)
            self.y = train["large_error"]

    def fit_model(self):
        """
        Fits the LightGBM classifier model.
        """
        # fixed parameters
        lgbm_params = {
            "random_seed": 42,
            "objective": "binary",
            "boosting_type": "gbdt",
            "verbosity": -1,
        }
        # grid for tuning
        param_grid = {
            "learning_rate": [0.02, 0.03, 0.04, 0.05],
            "num_leaves": [8, 16, 32, 64],
            "max_depth": [5, 10, 15],
            "n_estimators": [50, 100, 150],
        }

        lgbm = lgb.LGBMClassifier(**lgbm_params)

        # cv = Holdout(n=self.X.shape[0])
        grid_search = GridSearchCV(
            estimator=lgbm,
            param_grid=param_grid,
            scoring="roc_auc",
            n_jobs=-1,
        )
        grid_search.fit(self.X, self.y)

        self.classifier = grid_search.best_estimator_
