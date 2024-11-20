import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from tsfeatures import tsfeatures
from utils.MAST import MAST
from utils.PrepareDataset import PrepareDataset
from utils.BaselineModel import BaselineModel
from utils.ForecastingModel import ForecastingModel
from utils.MetaModel import MetaModel
from utils.TimeSeriesGenerator import TimeSeriesGenerator

from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.isotonic import IsotonicRegression


def balance_large_error(df, prefix):
    df_0 = df.query("large_error == 0")
    df_1 = df.query("large_error == 1")

    df_1_SYN = df_1[df_1["unique_id"].str.contains(prefix)]

    df_1_sampled = df_1_SYN.sample(n=len(df_0), random_state=42)

    balanced_df = (
        pd.concat([df_0, df_1_sampled])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    return balanced_df


# ____________________________________________________
def main(data, group, horizon, models):
    # data preparation

    dataset = PrepareDataset(dataset=data, group=group)
    dataset.load_dataset()
    dataset.train_test_valid_dev_split(horizon=horizon)

    if data == "M4":
        frequency = 1
    else:
        frequency = dataset.frequency
    seasonality = dataset.seasonality

    print(
        f"({data} {group}) frequency: {frequency}, seasonality: {seasonality}, horizon: {horizon}"
    )

    dev_set = dataset.dev_set
    valid = dataset.valid
    train = dataset.train
    test = dataset.test

    dev_set["unique_id"] = dev_set["unique_id"].astype("category")
    valid["unique_id"] = valid["unique_id"].astype(dev_set["unique_id"].dtype)
    train["unique_id"] = train["unique_id"].astype("category")
    test["unique_id"] = test["unique_id"].astype(train["unique_id"].dtype)

    # phase I

    # Forecast using Baseline Model

    baseline_model = BaselineModel(
        past_df=dev_set, seasonality=seasonality, frequency=frequency, horizon=horizon
    )

    # Train Forecasting model

    fm = ForecastingModel(
        models=models,
        train_set=dev_set,
        frequency=frequency,
        horizon=horizon,
        seasonality=seasonality,
    )
    fm.forecast(level=80)

    # MAST: main pipeline for meta-model training and model evaluation
    MAST_dev = MAST(
        test_set=valid,
        model_predictions=[
            fm.prediction,
            baseline_model.prediction,
        ],
        metrics=["smape"],
        seasonality=dataset.seasonality,
    )

    MAST_dev.evaluate_forecasts(train_set=dev_set)
    # ensure forecasting model(s) SMAPE score is lower
    evaluation = "\n".join(
        f"{model}: {(MAST_dev.error_summary[model].values[0] * 100):.2f}% SMAPE"
        for model in MAST_dev.error_summary.columns.drop("metric").to_list()
    )
    print(f"\n{evaluation}\n")

    best_model = MAST_dev.select_best()
    if best_model == "SeasonalNaive":
        raise ValueError(
            "Error: 'SeasonalNaive' is not a valid selection for processing."
        )

    MAST_dev.get_large_errors(model=best_model, metric="smape", quantile=0.80)
    datafile = data + "_" + group + ".csv"
    MAST_dev.extract_features(train_set=dev_set, filename=datafile)

    # metamodel data preparation

    mm_full = MAST_dev.features_errors.copy()
    mm_full.set_index("unique_id", inplace=True)

    cols_to_drop = [
        "large_error",
        "large_uncertainty",
        "class",
    ]

    # to the labels list, add the models information
    cols_to_drop += MAST_dev.error_summary.columns.to_list()

    # 70% for training metamodels, 30% for calibrating
    split_index = int(len(mm_full) * 0.7)

    mm_train = mm_full.iloc[:split_index]
    mm_cal = mm_full.iloc[split_index:]

    # metamodel1 - No resampling

    # set seed
    np.random.seed(42)

    metamodel1 = MetaModel(
        train_set=mm_train,
        model="LGBM",
        columns_to_drop=cols_to_drop,
    )

    # metamodel2 - ADASYN

    # set seed
    np.random.seed(42)

    metamodel2 = MetaModel(
        train_set=mm_train,
        model="LGBM",
        columns_to_drop=cols_to_drop,
        resampler="ADASYN",
    )

    # metamodel3 - SMOTE

    # set seed
    np.random.seed(42)

    metamodel3 = MetaModel(
        train_set=mm_train,
        model="LGBM",
        columns_to_drop=cols_to_drop,
        resampler="SMOTE",
    )

    # metamodel4 - BorderlineSMOTE

    # set seed
    np.random.seed(42)

    metamodel4 = MetaModel(
        train_set=mm_train,
        model="LGBM",
        columns_to_drop=cols_to_drop,
        resampler="BorderlineSMOTE",
    )

    # metamodel5 - SVMSMOTE

    # set seed
    np.random.seed(42)

    metamodel5 = MetaModel(
        train_set=mm_train,
        model="LGBM",
        columns_to_drop=cols_to_drop,
        resampler="SVMSMOTE",
    )

    # metamodel6 - RandomUnderSampler

    # set seed
    np.random.seed(42)

    metamodel6 = MetaModel(
        train_set=mm_train,
        model="LGBM",
        columns_to_drop=cols_to_drop,
        resampler="RandomUnderSampler",
    )

    prefix = "SYN"
    # metamodel7 - TSMixup
    # set seed
    np.random.seed(42)

    mixup_set = dev_set.copy()
    # remove series present in the calibration set
    mixup_set = mixup_set[~mixup_set["unique_id"].isin(mm_cal.index)]
    mm_train_large_errors_ids = [
        id for id in MAST_dev.large_errors_ids if id in mm_train.index
    ]

    mixup_set["large_error"] = mixup_set["unique_id"].apply(
        lambda x: 1 if x in mm_train_large_errors_ids else 0
    )
    mixup_set["unique_id"] = mixup_set["unique_id"].cat.remove_unused_categories()

    min_len = mixup_set["unique_id"].value_counts().min()
    max_len = mixup_set["unique_id"].value_counts().max()

    augmenter = TimeSeriesGenerator(
        mixup_set,
        seasonality=seasonality,
        frequency=frequency,
        min_len=min_len,
        max_len=max_len,
    )

    augmented_df = augmenter.generate_synthetic_dataset(
        method_name="TSMixup", n_samples=30
    )

    augmented_df["large_error"] = 1

    # to the generated series (+ original "large error" instances) re-add the remaining timeseries
    mix_df = pd.concat(
        [
            mixup_set[mixup_set["large_error"] == 0].reset_index(drop=True),
            augmented_df.reset_index(drop=True),
        ],
        ignore_index=True,
    )
    mix_df = mix_df.drop(columns="large_error")

    # extract features from original + generated series to train a metamodel

    features = tsfeatures(mix_df, freq=seasonality)
    features["large_error"] = features["unique_id"].apply(
        lambda x: 1 if x in MAST_dev.large_errors_ids or prefix in x else 0
    )
    features = balance_large_error(features, prefix)
    features = features.fillna(0)

    metamodel7 = MetaModel(
        train_set=features,
        model="LGBM",
        columns_to_drop=cols_to_drop,
    )

    # metamodel8 - DBA
    # set seed
    np.random.seed(42)
    augmented_df = augmenter.generate_synthetic_dataset(method_name="DBA", n_samples=30)

    augmented_df["large_error"] = 1

    # to the generated series (+ original "large error" instances) re-add the remaining timeseries
    mix_df = pd.concat(
        [
            mixup_set[mixup_set["large_error"] == 0].reset_index(drop=True),
            augmented_df.reset_index(drop=True),
        ],
        ignore_index=True,
    )
    mix_df = mix_df.drop(columns="large_error")

    # extract features from original + generated series to train a metamodel

    features = tsfeatures(mix_df, freq=seasonality)
    features["large_error"] = features["unique_id"].apply(
        lambda x: 1 if x in MAST_dev.large_errors_ids or prefix in x else 0
    )
    features = balance_large_error(features, prefix)
    features = features.fillna(0)

    metamodel8 = MetaModel(
        train_set=features,
        model="LGBM",
        columns_to_drop=cols_to_drop,
    )

    # metamodel9 - Jittering
    # set seed
    np.random.seed(42)
    augmented_df = augmenter.generate_synthetic_dataset(
        method_name="Jittering", n_samples=30
    )

    augmented_df["large_error"] = 1

    # to the generated series (+ original "large error" instances) re-add the remaining timeseries
    mix_df = pd.concat(
        [
            mixup_set[mixup_set["large_error"] == 0].reset_index(drop=True),
            augmented_df.reset_index(drop=True),
        ],
        ignore_index=True,
    )
    mix_df = mix_df.drop(columns="large_error")

    # extract features from original + generated series to train a metamodel

    features = tsfeatures(mix_df, freq=seasonality)
    features["large_error"] = features["unique_id"].apply(
        lambda x: 1 if x in MAST_dev.large_errors_ids or prefix in x else 0
    )
    features = balance_large_error(features, prefix)
    features = features.fillna(0)

    metamodel9 = MetaModel(
        train_set=features,
        model="LGBM",
        columns_to_drop=cols_to_drop,
    )

    # metamodel10 - Scaling
    # set seed
    np.random.seed(42)
    augmented_df = augmenter.generate_synthetic_dataset(
        method_name="Scaling", n_samples=30
    )

    augmented_df["large_error"] = 1

    # to the generated series (+ original "large error" instances) re-add the remaining timeseries
    mix_df = pd.concat(
        [
            mixup_set[mixup_set["large_error"] == 0].reset_index(drop=True),
            augmented_df.reset_index(drop=True),
        ],
        ignore_index=True,
    )
    mix_df = mix_df.drop(columns="large_error")

    # extract features from original + generated series to train a metamodel

    features = tsfeatures(mix_df, freq=seasonality)
    features["large_error"] = features["unique_id"].apply(
        lambda x: 1 if x in MAST_dev.large_errors_ids or prefix in x else 0
    )
    features = balance_large_error(features, prefix)
    features = features.fillna(0)

    metamodel10 = MetaModel(
        train_set=features,
        model="LGBM",
        columns_to_drop=cols_to_drop,
    )

    # phase II

    # Forecast using Baseline Model

    baseline_model2 = BaselineModel(
        past_df=train, seasonality=seasonality, frequency=frequency, horizon=horizon
    )

    # Train Forecasting model

    fm2 = ForecastingModel(
        models=models,
        train_set=train,
        frequency=frequency,
        horizon=horizon,
        seasonality=seasonality,
    )
    fm2.forecast(level=80)

    # MAST: main pipeline for meta-model training and model evaluation
    mast = MAST(
        test_set=test,
        model_predictions=[
            fm2.prediction,
            baseline_model2.prediction,
        ],
        metrics=["smape"],
        seasonality=dataset.seasonality,
    )

    mast.evaluate_forecasts(train_set=train)

    # ensure forecasting model surpasses baseline again
    evaluation = "\n".join(
        f"{model} II: {(mast.error_summary[model].values[0] * 100):.2f}% SMAPE"
        for model in mast.error_summary.columns.drop("metric").to_list()
    )
    print(f"\n{evaluation}\n")

    best_model = mast.select_best()
    if best_model == "SeasonalNaive":
        raise ValueError(
            "Error: 'SeasonalNaive' is not a valid selection for processing."
        )

    mast.get_large_errors(quantile=0.80, model=best_model, metric="smape")
    datafile = data + "_" + group + ".csv"
    mast.extract_features(train_set=train, filename=datafile)

    # Features and large_errors from the full "train" data, to test metamodels

    full_features_df = mast.features_errors.copy()
    full_features_df.set_index("unique_id", inplace=True)
    full_features_df.drop(
        columns=mast.error_summary.columns.drop("SeasonalNaive").to_list(), inplace=True
    )
    full_features_df.fillna(0, inplace=True)

    X_t = full_features_df.drop(["large_error"], axis=1)
    y_t = full_features_df["large_error"]

    # Inference

    # calibration using isotonic regression

    X_cal = mm_cal.drop([col for col in cols_to_drop if col in mm_cal.columns], axis=1)
    y_cal = mm_cal["large_error"]

    # check if there is leakage from train and calibration data

    overlap_indices = mm_train.index.intersection(mm_cal.index)
    if not overlap_indices.empty:
        raise ValueError(
            f"Overlap found in {len(overlap_indices)} indices between training and calibration sets: {overlap_indices}"
        )

    # calibration

    # fit isotonic regressions to the raw probabilities of each metamodel

    oversamplers = ["ADASYN", "SMOTE", "BorderlineSMOTE", "SVMSMOTE"]
    undersamplers = ["RandomUnderSampler"]
    generators = ["TSMixup", "DBA", "Jittering", "Scaling"]
    models = oversamplers + undersamplers + generators + ["No Resampling"]

    meta_models = [
        metamodel2,
        metamodel3,
        metamodel4,
        metamodel5,
        metamodel6,
        metamodel7,
        metamodel8,
        metamodel9,
        metamodel10,
        metamodel1,
    ]

    calibrated_models = []

    for i, mm in enumerate(meta_models):
        # metamodel raw predicted probabilities
        y_prob_before = mm.classifier.predict_proba(X_cal)[
            :, 1
        ]  # only for positive class

        isotonic = IsotonicRegression(out_of_bounds="clip")
        isotonic.fit_transform(y_prob_before, y_cal)

        calibrated_models.append(isotonic)

    # calibrated isotonic models on the test set

    def calibrated_predict(X, y):
        log_losses_t = []
        for i, mm in enumerate(meta_models):
            y_prob_before = mm.classifier.predict_proba(X)[:, 1]
            y_prob_calibrated = calibrated_models[i].transform(y_prob_before)

            log_losses_t.append(log_loss(y, y_prob_calibrated))

        return log_losses_t

    log_losses = calibrated_predict(X_t, y_t)
    sorted_results = sorted(zip(models, log_losses), key=lambda x: x[1])

    print("\nCalibrated Predictions:\n")
    for model, l_loss in sorted_results:
        if model in oversamplers:
            color = "\033[92m"  # Green for oversamplers
        elif model in undersamplers:
            color = "\033[91m"  # Red for undersamplers
        elif model in generators:
            color = "\033[93m"  # Orange (Yellow) for generators
        else:
            color = "\033[0m"  # Default color for others

        print(f"{color}{model}\033[0m Log Loss: {l_loss:.3f}")
    print()


# Save the evaluation results
# results = {}
# for i in range(len(models)):
#     results[models[i]] = {
#         "ROC AUC": round(roc_aucs[i], 3),
#         "Log Loss": round(log_losses[i], 3),
#         "Brier Score": round(brier_scores[i], 3),
#     }

# os.makedirs(f"results_q80_{data}_{group}", exist_ok=True)
# with open(
#     f"results_q80_{data}_{group}/calibration_{oversampler}_{generator}.txt", "w"
# ) as f:
#     json.dump(results, f, indent=4)

# print(f"\nEvaluation results saved to 'calibration_{oversampler}_{generator}.txt'")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="MAST resampling algorithm testing pipeline"
    )

    parser.add_argument(
        "--data",
        dest="data",
        type=str,
        default="M3",
        help="Specify the dataset to use",
    )

    parser.add_argument(
        "--group",
        dest="group",
        type=str,
        default="Monthly",
        help="Specify the sampling frequency to use",
    )

    parser.add_argument(
        "--horizon",
        dest="horizon",
        type=int,
        default=12,
        help="Specify the forecasting horizon to use",
    )

    parser.add_argument(
        "--models",
        nargs="*",
        type=str,
        default=["LGBM", "XGB"],
        help="A forecasting model or a list of forecasting models.",
    )

    args = parser.parse_args()
    if args.models and len(args.models) == 1:
        args.models = args.models[0]
    main(
        data=args.data,
        group=args.group,
        horizon=args.horizon,
        models=args.models,
    )
