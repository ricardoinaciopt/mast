import os
import gc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import shap
import lightgbm as lgb
from utils.PrepareDataset import PrepareDataset
from utils.BaselineModel import BaselineModel
from utils.ForecastingModel import ForecastingModel
from utils.MAST import MAST
from utils.MetaModel import MetaModel


def main(resampler):

    dataset = PrepareDataset(dataset="Tourism", group="Monthly")
    dataset.load_dataset()
    horizon = 12
    dataset.train_test_valid_dev_split(horizon=horizon)

    train = dataset.train
    test = dataset.test
    dev_set = dataset.dev_set
    valid = dataset.valid

    # Phase 1 - Use Deveopment Data
    # Train Baseline Model

    baseline_model = BaselineModel(
        past_df=dev_set, seasonality=dataset.seasonality, frequency=dataset.frequency
    )
    baseline_model.forecast(horizon=horizon)

    # Train LightGBM
    forecastingModel = ForecastingModel(
        frequency=dataset.frequency,
        horizon=horizon,
        lags=[(i + 1) for i in range(12)],
        train=dev_set,
    )

    forecastingModel.train()
    forecastingModel.forecast()

    # MAST: main pipeline for meta-model training and model evaluation
    mast_dev = MAST(
        test_set=valid,
        model_predictions=[forecastingModel.prediction, baseline_model.prediction],
        models=["SeasonalNaive", "ForecastingModel"],
        seasonality=dataset.seasonality,
    )
    mast_dev.evaluate_forecasts(train_set=dev_set)
    error_summary1 = mast_dev.summary.to_dict(orient="records")

    mast_dev.get_large_errors(0.95, "ForecastingModel", "smape")
    mast_dev.extract_features(train_set=dev_set, frequency=12)

    # Fit the meta model
    metamodel1 = MetaModel(
        model="ForecastingModel", train_set=mast_dev.features_errors.copy()
    )
    metamodel1.fit_model()

    # Fit the resampled meta model
    metamodel2 = MetaModel(
        model="ForecastingModel",
        train_set=mast_dev.features_errors.copy(),
        resampler=resampler,
    )
    metamodel2.fit_model()

    # Phase II - Use Whole Data
    # Train 2nd Baseline Model

    baseline_model2 = BaselineModel(
        past_df=train, seasonality=dataset.seasonality, frequency=dataset.frequency
    )
    baseline_model2.forecast(horizon=horizon)

    # Train 2nd LightGBM
    forecastingModel2 = ForecastingModel(
        frequency=dataset.frequency,
        horizon=horizon,
        lags=[(i + 1) for i in range(12)],
        train=train,
    )

    forecastingModel2.train()
    forecastingModel2.forecast()

    # MAST: main pipeline for meta-model training and model evaluation
    mast = MAST(
        test_set=test,
        model_predictions=[forecastingModel2.prediction, baseline_model2.prediction],
        models=["SeasonalNaive", "ForecastingModel"],
        seasonality=dataset.seasonality,
    )
    mast.evaluate_forecasts(train_set=train)
    error_summary2 = mast.summary.to_dict(orient="records")

    mast.get_large_errors(0.95, "ForecastingModel", "smape")
    mast.extract_features(train_set=train, frequency=12)

    full_features_df = mast.features_errors.copy()
    full_features_df.set_index("unique_id", inplace=True)
    full_features_df.drop(columns=["metric", "ForecastingModel"], inplace=True)
    full_features_df.fillna(0, inplace=True)
    full_features_df.head()
    X = full_features_df.drop(["large_error"], axis=1)
    y = full_features_df["large_error"]

    mm1_eval = metamodel1.classifier.predict(X)
    mm2_eval = metamodel2.classifier.predict(X)

    roc_auc_final_1 = roc_auc_score(y, mm1_eval)
    print("AUC score for LGBM (Meta Model 1):", roc_auc_final_1)

    roc_auc_final_2 = roc_auc_score(y, mm2_eval)
    print("AUC score for LGBM (Meta Model 2):", roc_auc_final_2)

    if not os.path.exists("figs"):
        os.makedirs("figs")

    # Feature Importances (augmented metamodel)
    fig = plt.figure(figsize=(10, 7))
    lgb.plot_importance(
        metamodel2.classifier,
        ax=fig.gca(),
        max_num_features=10,
        grid=False,
        dpi=1000,
    )
    importance_file = "figs/features_importances_" + str(resampler) + "_Tourism.pdf"
    plt.savefig(importance_file, dpi=1000, bbox_inches="tight")
    plt.close(fig)

    # SHAP values (augmented metamodel)
    explainer = shap.TreeExplainer(metamodel2.classifier)
    shap_values = explainer(X)
    fig = plt.figure()
    shap.summary_plot(shap_values, X, max_display=10, show=False)
    shap_file = "figs/shap_" + str(resampler) + "_Tourism.pdf"
    plt.savefig(shap_file, dpi=1000, bbox_inches="tight")
    plt.close(fig)

    del shap_values
    del explainer
    del fig
    gc.collect()

    return (
        error_summary1,
        error_summary2,
        roc_auc_final_1,
        roc_auc_final_2,
    )
