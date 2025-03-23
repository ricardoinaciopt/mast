import os
import json
import argparse
import numpy as np
import pandas as pd
from tsfeatures import tsfeatures
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from utils.MAST import MAST
from utils.MetaModel import MetaModel
from utils.BaselineModel import BaselineModel
from utils.PrepareDataset import PrepareDataset
from utils.ForecastingModel import ForecastingModel
from utils.TimeSeriesGenerator import TimeSeriesGenerator


def balance_large_error(df, prefix):
    df_0 = df.query("large_error == 0")
    df_1 = df.query("large_error == 1")

    df_1_with_prefix = df_1[df_1["unique_id"].str.contains(prefix, na=False)]

    df_1_without_prefix = df_1[~df_1["unique_id"].str.contains(prefix, na=False)]

    if len(df_1_with_prefix) < len(df_0):
        raise ValueError("Not enough syntehtic samples to balance the dataset.")

    # sample synthetic series to match the number of non-large errors + non synthetic large errors
    df_1_sampled = df_1_with_prefix.sample(
        n=(len(df_0) - len(df_1_without_prefix)), random_state=42
    )

    balanced_df_1 = pd.concat([df_1_without_prefix, df_1_sampled])

    # balanced dataset (len(large_error==0) == len(large_error==1))
    balanced_df = (
        pd.concat([df_0, balanced_df_1])
        .sample(frac=1, random_state=42)  # shuffle dataset
        .reset_index(drop=True)
    )

    return balanced_df


# ____________________________________________________
def main(data, group, horizon, models, quantile, level):
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

    # convert unique_id to categorical
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
        prediction_intervals=True,
    )
    fm.forecast(level=level)

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

    MAST_dev.get_large_errors(model=best_model, metric="smape", quantile=quantile)

    datafile = data + "_" + group + ".csv"
    MAST_dev.extract_features(train_set=dev_set, filename=datafile)

    MAST_dev.compute_uncertainty(
        train_set=dev_set, predictions=MAST_dev.merged_forecasts, level=level
    )
    MAST_dev.get_large_uncertainty(model="LGBM", quantile=quantile)
    features_errors_uncertainty_dev = MAST_dev.features_errors.copy()

    features_errors_uncertainty_dev["large_uncertainty"] = (
        features_errors_uncertainty_dev["unique_id"].apply(
            lambda x: 1 if x in MAST_dev.large_uncertainty_ids else 0
        )
    )

    conditions = [
        (features_errors_uncertainty_dev["large_error"] == 0)
        & (
            features_errors_uncertainty_dev["large_uncertainty"] == 0
        ),  # 0 - no stress (none)
        (features_errors_uncertainty_dev["large_error"] == 1)
        & (
            features_errors_uncertainty_dev["large_uncertainty"] == 0
        ),  # 1 - large error
        (features_errors_uncertainty_dev["large_error"] == 0)
        & (
            features_errors_uncertainty_dev["large_uncertainty"] == 1
        ),  # 2 - large uncertainty
        (features_errors_uncertainty_dev["large_error"] == 1)
        & (
            features_errors_uncertainty_dev["large_uncertainty"] == 1
        ),  # 3 - very stressed (both)
    ]

    choices = [0, 1, 2, 3]

    features_errors_uncertainty_dev["class"] = np.select(conditions, choices)

    # metamodel data preparation

    mm_full = features_errors_uncertainty_dev.copy()
    mm_full.set_index("unique_id", inplace=True)

    cols_to_drop = [
        "large_error",
        "error_quantile_LGBM",
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
        target="error",
    )

    # metamodel2 - ADASYN

    # set seed
    np.random.seed(42)

    metamodel2 = MetaModel(
        train_set=mm_train,
        model="LGBM",
        columns_to_drop=cols_to_drop,
        resampler="ADASYN",
        target="error",
    )

    # metamodel3 - SMOTE

    # set seed
    np.random.seed(42)

    metamodel3 = MetaModel(
        train_set=mm_train,
        model="LGBM",
        columns_to_drop=cols_to_drop,
        resampler="SMOTE",
        target="error",
    )

    # metamodel4 - BorderlineSMOTE

    # set seed
    np.random.seed(42)

    metamodel4 = MetaModel(
        train_set=mm_train,
        model="LGBM",
        columns_to_drop=cols_to_drop,
        resampler="BorderlineSMOTE",
        target="error",
    )

    # metamodel5 - SVMSMOTE

    # set seed
    np.random.seed(42)

    metamodel5 = MetaModel(
        train_set=mm_train,
        model="LGBM",
        columns_to_drop=cols_to_drop,
        resampler="SVMSMOTE",
        target="error",
    )

    # metamodel6 - RandomUnderSampler

    # set seed
    np.random.seed(42)

    metamodel6 = MetaModel(
        train_set=mm_train,
        model="LGBM",
        columns_to_drop=cols_to_drop,
        resampler="RandomUnderSampler",
        target="error",
    )

    # prefix to add to the generated time series
    prefix = "SYN"

    # metamodel7 - TSMixup
    # set seed
    np.random.seed(42)

    gen_set = dev_set.copy()
    # remove series present in the calibration set
    gen_set = gen_set[~gen_set["unique_id"].isin(mm_cal.index)]
    mm_train_large_error_ids = [
        id for id in MAST_dev.large_errors_ids if id in mm_train.index
    ]

    gen_set["large_error"] = gen_set["unique_id"].apply(
        lambda x: 1 if x in mm_train_large_error_ids else 0
    )
    gen_set["unique_id"] = gen_set["unique_id"].cat.remove_unused_categories()

    min_len = gen_set["unique_id"].value_counts().min()
    max_len = gen_set["unique_id"].value_counts().max()

    # define the synthetic data generator wrapper
    augmenter = TimeSeriesGenerator(
        gen_set,
        dataset=data,
        group=group,
        seasonality=seasonality,
        frequency=frequency,
        min_len=min_len,
        max_len=max_len,
        target="errors",
    )

    synthetic_ts_df = augmenter.generate_synthetic_dataset(method_name="TSMixup")

    synthetic_ts_df["large_error"] = 1

    # to the generated syntehtic series re-add the original timeseries
    gen_df = pd.concat(
        [
            gen_set.reset_index(drop=True),
            synthetic_ts_df.reset_index(drop=True),
        ],
        ignore_index=True,
    )

    if gen_df.duplicated().any():
        raise ValueError("The synthetically augmented dataset has repeated values.")

    # extract features from original + generated series to train a metamodel

    gen_df = gen_df.drop(columns="large_error")

    features = tsfeatures(gen_df, freq=seasonality)
    features["large_error"] = features["unique_id"].apply(
        lambda x: 1 if x in mm_train_large_error_ids or prefix in x else 0
    )
    features = balance_large_error(features, prefix)
    features = features.fillna(0)

    metamodel7 = MetaModel(
        train_set=features,
        model="LGBM",
        columns_to_drop=cols_to_drop,
        target="error",
    )

    # metamodel8 - DBA
    # set seed
    np.random.seed(42)
    synthetic_ts_df = augmenter.generate_synthetic_dataset(method_name="DBA")

    synthetic_ts_df["large_error"] = 1

    # to the generated syntehtic series re-add the original timeseries
    gen_df = pd.concat(
        [
            gen_set.reset_index(drop=True),
            synthetic_ts_df.reset_index(drop=True),
        ],
        ignore_index=True,
    )
    # extract features from original + generated series to train a metamodel

    gen_df = gen_df.drop(columns="large_error")

    features = tsfeatures(gen_df, freq=seasonality)
    features["large_error"] = features["unique_id"].apply(
        lambda x: 1 if x in mm_train_large_error_ids or prefix in x else 0
    )
    features = balance_large_error(features, prefix)
    features = features.fillna(0)

    metamodel8 = MetaModel(
        train_set=features,
        model="LGBM",
        columns_to_drop=cols_to_drop,
        target="error",
    )

    # metamodel9 - Jittering
    # set seed
    np.random.seed(42)
    synthetic_ts_df = augmenter.generate_synthetic_dataset(method_name="Jittering")

    synthetic_ts_df["large_error"] = 1

    # to the generated syntehtic series re-add the original timeseries

    gen_df = pd.concat(
        [
            gen_set.reset_index(drop=True),
            synthetic_ts_df.reset_index(drop=True),
        ],
        ignore_index=True,
    )
    # extract features from original + generated series to train a metamodel

    gen_df = gen_df.drop(columns="large_error")

    features = tsfeatures(gen_df, freq=seasonality)
    features["large_error"] = features["unique_id"].apply(
        lambda x: 1 if x in mm_train_large_error_ids or prefix in x else 0
    )
    features = balance_large_error(features, prefix)
    features = features.fillna(0)

    metamodel9 = MetaModel(
        train_set=features,
        model="LGBM",
        columns_to_drop=cols_to_drop,
        target="error",
    )

    # metamodel11 - Scaling
    # set seed
    np.random.seed(42)
    synthetic_ts_df = augmenter.generate_synthetic_dataset(method_name="Scaling")

    synthetic_ts_df["large_error"] = 1

    # to the generated syntehtic series re-add the original timeseries
    gen_df = pd.concat(
        [
            gen_set.reset_index(drop=True),
            synthetic_ts_df.reset_index(drop=True),
        ],
        ignore_index=True,
    )
    # extract features from original + generated series to train a metamodel

    gen_df = gen_df.drop(columns="large_error")

    features = tsfeatures(gen_df, freq=seasonality)
    features["large_error"] = features["unique_id"].apply(
        lambda x: 1 if x in mm_train_large_error_ids or prefix in x else 0
    )
    features = balance_large_error(features, prefix)
    features = features.fillna(0)

    metamodel11 = MetaModel(
        train_set=features,
        model="LGBM",
        columns_to_drop=cols_to_drop,
        target="error",
    )

    # metamodel12 - MagnitudeWarping
    # set seed
    np.random.seed(42)
    synthetic_ts_df = augmenter.generate_synthetic_dataset(
        method_name="MagnitudeWarping"
    )

    synthetic_ts_df["large_error"] = 1

    # to the generated syntehtic series re-add the original timeseries
    gen_df = pd.concat(
        [
            gen_set.reset_index(drop=True),
            synthetic_ts_df.reset_index(drop=True),
        ],
        ignore_index=True,
    )
    # extract features from original + generated series to train a metamodel

    gen_df = gen_df.drop(columns="large_error")

    features = tsfeatures(gen_df, freq=seasonality)
    features["large_error"] = features["unique_id"].apply(
        lambda x: 1 if x in mm_train_large_error_ids or prefix in x else 0
    )
    features = balance_large_error(features, prefix)
    features = features.fillna(0)

    metamodel12 = MetaModel(
        train_set=features,
        model="LGBM",
        columns_to_drop=cols_to_drop,
        target="error",
    )

    # metamodel13 - TimeWarping
    # set seed
    np.random.seed(42)
    synthetic_ts_df = augmenter.generate_synthetic_dataset(method_name="TimeWarping")

    synthetic_ts_df["large_error"] = 1

    # to the generated syntehtic series re-add the original timeseries
    gen_df = pd.concat(
        [
            gen_set.reset_index(drop=True),
            synthetic_ts_df.reset_index(drop=True),
        ],
        ignore_index=True,
    )
    # extract features from original + generated series to train a metamodel

    gen_df = gen_df.drop(columns="large_error")

    features = tsfeatures(gen_df, freq=seasonality)
    features["large_error"] = features["unique_id"].apply(
        lambda x: 1 if x in mm_train_large_error_ids or prefix in x else 0
    )
    features = balance_large_error(features, prefix)
    features = features.fillna(0)

    metamodel13 = MetaModel(
        train_set=features,
        model="LGBM",
        columns_to_drop=cols_to_drop,
        target="error",
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
        prediction_intervals=True,
    )
    fm2.forecast(level=level)

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

    folder_prefix = "q" + str(int(quantile * 100)) + "_" + str(level)
    os.makedirs(f"lgbm_results/{folder_prefix}/le", exist_ok=True)
    with open(
        f"lgbm_results/{folder_prefix}/le/{folder_prefix}_{data}_{group}.txt", "w"
    ) as f:
        f.write(str(evaluation))

    best_model = mast.select_best()
    if best_model == "SeasonalNaive":
        raise ValueError(
            "Error: 'SeasonalNaive' is not a valid selection for processing."
        )

    mast.get_large_errors(quantile=quantile, model=best_model, metric="smape")
    datafile = data + "_" + group + ".csv"
    mast.extract_features(train_set=train, filename=datafile)

    mast.compute_uncertainty(
        train_set=dev_set, predictions=mast.merged_forecasts, level=level
    )
    mast.get_large_uncertainty(model="LGBM", quantile=quantile)

    features_errors_uncertainty = mast.features_errors.copy()
    features_errors_uncertainty["large_uncertainty"] = features_errors_uncertainty[
        "unique_id"
    ].apply(lambda x: 1 if x in mast.large_uncertainty_ids else 0)

    conditions = [
        (features_errors_uncertainty["large_error"] == 0)
        & (
            features_errors_uncertainty["large_uncertainty"] == 0
        ),  # 0 - no stress (none)
        (features_errors_uncertainty["large_error"] == 1)
        & (features_errors_uncertainty["large_uncertainty"] == 0),  # 1 - large error
        (features_errors_uncertainty["large_error"] == 0)
        & (
            features_errors_uncertainty["large_uncertainty"] == 1
        ),  # 2 - large uncertainty
        (features_errors_uncertainty["large_error"] == 1)
        & (
            features_errors_uncertainty["large_uncertainty"] == 1
        ),  # 3 - very stressed (both)
    ]

    choices = [0, 1, 2, 3]

    features_errors_uncertainty["class"] = np.select(conditions, choices)

    # Features and large_errors from the full "train" data, to test metamodels

    full_features_df = features_errors_uncertainty.copy()
    full_features_df.set_index("unique_id", inplace=True)
    full_features_df.drop(
        columns=mast.error_summary.columns.drop("SeasonalNaive").to_list(), inplace=True
    )
    full_features_df.fillna(0, inplace=True)

    # Save data for errors and uncertainty analysis.
    err_unc_file = os.path.join("errors_uncertainty", f"{data}_{group}_full.csv")

    if not os.path.exists(err_unc_file):
        os.makedirs("errors_uncertainty", exist_ok=True)
        mm_full.to_csv(err_unc_file)

    X_t = full_features_df.drop(
        ["large_uncertainty", "error_quantile_LGBM", "large_error", "class"], axis=1
    )
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
    generators = [
        "TSMixup",
        "DBA",
        "Jittering",
        "Scaling",
        "MagnitudeWarping",
        "TimeWarping",
    ]
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
        metamodel11,
        metamodel12,
        metamodel13,
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
        roc_auc_t, log_losses_t, brier_scores_t = [], [], []
        unique_ids = X.index.unique()
        results_by_id = {uid: {} for uid in unique_ids}

        for i, mm in enumerate(meta_models):
            y_prob_before = mm.classifier.predict_proba(X)[:, 1]
            y_prob_calibrated = calibrated_models[i].transform(y_prob_before)

            roc_auc_t.append(roc_auc_score(y, y_prob_calibrated))
            log_losses_t.append(log_loss(y, y_prob_calibrated))
            brier_scores_t.append(brier_score_loss(y, y_prob_calibrated))

            for uid in unique_ids:
                mask = X.index == uid
                y_true_uid = y[mask]
                y_prob_calibrated_uid = y_prob_calibrated[mask]
                results_by_id[uid][models[i]] = {
                    "log_loss": log_loss(
                        y_true_uid, y_prob_calibrated_uid, labels=[0, 1]
                    )
                }

        return roc_auc_t, log_losses_t, brier_scores_t, results_by_id

    roc_aucs, log_losses, brier_scores, results_by_id = calibrated_predict(X_t, y_t)
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
    results = {
        "roc_aucs": {model: roc_auc for model, roc_auc in zip(models, roc_aucs)},
        "log_losses": {model: log_loss for model, log_loss in zip(models, log_losses)},
        "brier_scores": {
            model: brier_score for model, brier_score in zip(models, brier_scores)
        },
        "by_id": results_by_id,
    }

    os.makedirs(f"results/{folder_prefix}/le", exist_ok=True)
    with open(
        f"results/{folder_prefix}/le/{folder_prefix}_{data}_{group}.json", "w"
    ) as f:
        json.dump(results, f, indent=4)


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

    parser.add_argument(
        "--quantile",
        dest="quantile",
        type=float,
        default=0.8,
        help="Specify the quantile to use for large errors.",
    )

    parser.add_argument(
        "--level",
        dest="level",
        type=int,
        default=90,
        help="Specify the level to use for confidence intervals.",
    )

    args = parser.parse_args()
    if args.models and len(args.models) == 1:
        args.models = args.models[0]
    main(
        data=args.data,
        group=args.group,
        horizon=args.horizon,
        models=args.models,
        quantile=args.quantile,
        level=args.level,
    )
