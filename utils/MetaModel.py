import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

# oversampling
from imblearn.over_sampling import (
    SMOTE,
    ADASYN,
    SVMSMOTE,
    BorderlineSMOTE,
)

# undersampling
from imblearn.under_sampling import RandomUnderSampler


class MetaModel:
    """
    MetaModel class for training a meta-model using LightGBM or XGBoost with optional data resampling.

    Attributes:
        train (pd.DataFrame): Training dataset.
        model (str): Name of the model.
        resampler (str): Name of the data resampling technique (default=None).
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
        resamplers (dict): Dictionary of resampling techniques.
        classifier (LGBMClassifier or XGBoostClassifier): The best estimator (classifier) after tuning a given model.

    Methods:
        preprocess_set(train): Preprocesses the training set, including data resampling if specified.
        fit_model(): Fits the classifier model, while performing hyperparameter tuning, if specified.
    """

    def __init__(self, train_set, model, columns_to_drop, resampler=None, tuning=True):
        """
        Initializes the MetaModel object.

        Args:
            train_set (pd.DataFrame): Training dataset.
            model (str): Name of the model.
            columns_to_drop (list): Features to drop from the train dataset.
            tuning (bool): Flag to specify if hyperparameter tuning should be conducted.
            resampler (str): Name of the data resampling technique (default=None).
        """
        self.model = model
        self.resampler = resampler
        self.tuning = tuning
        self.train = train_set.copy()
        self.columns_to_drop = columns_to_drop
        self.X = None
        self.y = None
        self.resamplers = {
            "SMOTE": SMOTE(),
            "ADASYN": ADASYN(),
            "SVMSMOTE": SVMSMOTE(),
            "BorderlineSMOTE": BorderlineSMOTE(),
            "RandomUnderSampler": RandomUnderSampler(),
        }
        self.classifier = None
        self.preprocess_set()
        self.fit_model()

    def preprocess_set(self):
        """
        Preprocesses the training set, including data resampling if specified.

        Args:
            train (pd.DataFrame): Training dataset.
        """
        if "unique_id" in self.train.columns:
            self.train.set_index("unique_id", inplace=True)
        if "metric" in self.train.columns:
            self.train.drop(
                columns=["metric"], inplace=True
            )  # remove this categorical feature first to allow resampling
        self.train.fillna(0, inplace=True)

        if self.resampler and self.resampler in self.resamplers:
            data_to_resample = self.train.copy()
            threshold_mask = self.train["large_error"].astype(int)

            resampled_data, err_class = self.resamplers[self.resampler].fit_resample(
                data_to_resample, threshold_mask
            )

            self.X = resampled_data.drop(
                [col for col in self.columns_to_drop if col in resampled_data.columns],
                axis=1,
            )
            self.y = resampled_data["large_error"]
        else:
            self.X = self.train.drop(
                [col for col in self.columns_to_drop if col in self.train.columns],
                axis=1,
            )
            self.y = self.train["large_error"]

    def fit_model(self):
        """
        Fits the classifier model.
        """
        lgbm_params = {"n_estimators": 200, "verbosity": -1}
        xgb_params = {"n_estimators": 200, "verbosity": 0}

        if self.model == "LGBM":
            param_grid = {
                "num_leaves": [3, 5, 10, 15],
                "max_depth": [-1, 3, 5, 10, 15],
                "lambda_l1": [0.1, 1, 10, 100],
                "lambda_l2": [0.1, 1, 10, 100],
                "learning_rate": [0.05, 0.1, 0.2],
                "min_child_samples": [7, 15, 30],
            }
            estimator = lgb.LGBMClassifier(**lgbm_params)
        elif self.model == "XGB":
            param_grid = {
                "max_depth": [3, 5, 10, 15],
                "lambda": [0.1, 1, 10, 100],
                "alpha": [0.1, 1, 10, 100],
                "learning_rate": [0.05, 0.1, 0.2],
                "min_child_weight": [7, 15, 30],
            }
            estimator = xgb.XGBClassifier(**xgb_params)
        else:
            print("Unsupported algorithm, use: LGBM or XGB.")
            return

        if self.tuning:
            rnd_search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_grid,
                scoring="roc_auc",
                n_jobs=-1,
            )
            print(f"Starting MetaModel {self.model} tuning and fitting.")
            rnd_search.fit(self.X, self.y)

            self.classifier = rnd_search.best_estimator_
        else:
            print(f"Starting MetaModel {self.model} fitting.")
            self.classifier = estimator.fit(self.X, self.y)
