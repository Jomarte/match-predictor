from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .data import load_processed_matches
from .features import RESULT_LABELS, build_feature_frame


XGB_DEFAULTS = {
    "learning_rate": 0.03,
    "max_depth": 3,
    "min_child_weight": 3,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 2.0,
}


def default_model_path(models_dir: Path, competition: str) -> Path:
    return models_dir / f"{competition.upper()}_model.joblib"


def default_metrics_path(models_dir: Path, competition: str) -> Path:
    return models_dir / f"{competition.upper()}_metrics.json"


def default_curve_path(models_dir: Path, competition: str) -> Path:
    return models_dir / f"{competition.upper()}_training_curve.csv"


def default_curve_plot_path(models_dir: Path, competition: str) -> Path:
    return models_dir / f"{competition.upper()}_training_curve.png"


def save_training_curve_plot(curve_path: Path, plot_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = pd.read_csv(curve_path)
    x_column = "iteration" if "iteration" in curve.columns else "n_estimators"
    test_loss_column = "test_mlogloss" if "test_mlogloss" in curve.columns else "test_log_loss"
    train_loss_column = "train_mlogloss" if "train_mlogloss" in curve.columns else None

    fig, loss_axis = plt.subplots(figsize=(9, 5))
    loss_axis.plot(
        curve[x_column],
        curve[test_loss_column],
        color="#1f77b4",
        label="Test mlogloss",
    )
    if train_loss_column:
        loss_axis.plot(
            curve[x_column],
            curve[train_loss_column],
            color="#ff7f0e",
            alpha=0.85,
            label="Train mlogloss",
        )
    loss_axis.set_xlabel("Iteration" if x_column == "iteration" else "Number of trees")
    loss_axis.set_ylabel("Multiclass log loss")
    loss_axis.grid(True, alpha=0.25)
    loss_axis.legend(loc="best")

    fig.suptitle("XGBoost training curve")
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def _build_training_dataset(
    processed_csv: Path, last_n: int
) -> tuple[pd.DataFrame, list[str]]:
    matches = load_processed_matches(processed_csv)
    features, labels, meta = build_feature_frame(
        matches, last_n=last_n, include_unfinished=False
    )
    dataset = pd.concat(
        [
            meta.reset_index(drop=True),
            labels.reset_index(drop=True).rename("label"),
            features.reset_index(drop=True),
        ],
        axis=1,
    )
    dataset = dataset.dropna(subset=["label", "season_start_year"])
    return dataset, list(features.columns)


def _fit_xgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    feature_columns: list[str],
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    use_gpu: bool,
    x_eval: pd.DataFrame | None = None,
    y_eval: pd.Series | None = None,
) -> tuple[Pipeline, LabelEncoder, str, dict[str, Any]]:
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train_encoded)
    classes = list(label_encoder.classes_)
    if len(classes) < 2:
        raise ValueError("Need at least two result classes to train a classifier.")

    imputer = SimpleImputer(strategy="median")
    x_train_imputed = imputer.fit_transform(x_train)
    eval_set = None
    if x_eval is not None and y_eval is not None:
        y_eval_encoded = label_encoder.transform(y_eval)
        x_eval_imputed = imputer.transform(x_eval)
        eval_set = [(x_train_imputed, y_train_encoded), (x_eval_imputed, y_eval_encoded)]

    device_used = "cuda" if use_gpu else "cpu"
    classifier = XGBClassifier(
        objective="multi:softprob",
        num_class=len(classes),
        eval_metric="mlogloss",
        tree_method="hist",
        device=device_used,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=XGB_DEFAULTS["subsample"],
        colsample_bytree=XGB_DEFAULTS["colsample_bytree"],
        min_child_weight=XGB_DEFAULTS["min_child_weight"],
        reg_lambda=XGB_DEFAULTS["reg_lambda"],
        random_state=42,
        n_jobs=-1,
    )

    fit_kwargs: dict[str, Any] = {
        "sample_weight": sample_weight,
        "verbose": False,
    }
    if eval_set is not None:
        fit_kwargs["eval_set"] = eval_set

    try:
        classifier.fit(x_train_imputed, y_train_encoded, **fit_kwargs)
    except Exception:
        if not use_gpu:
            raise
        device_used = "cpu"
        classifier.set_params(device="cpu")
        classifier.fit(x_train_imputed, y_train_encoded, **fit_kwargs)

    pipeline = Pipeline(
        steps=[
            ("imputer", imputer),
            ("classifier", classifier),
        ]
    )
    evals_result = classifier.evals_result() if eval_set is not None else {}
    return pipeline, label_encoder, device_used, evals_result


def _evaluate_model(
    model: Pipeline,
    label_encoder: LabelEncoder,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    classes = list(label_encoder.classes_)
    probabilities = model.predict_proba(x_test)
    predictions_encoded = model.predict(x_test).astype(int)
    predictions = label_encoder.inverse_transform(predictions_encoded)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "log_loss": float(log_loss(y_test, probabilities, labels=classes)),
        "classification_report": classification_report(
            y_test, predictions, labels=classes, output_dict=True, zero_division=0
        ),
    }


def _curve_rows_from_evals(evals_result: dict[str, Any]) -> list[dict[str, Any]]:
    train_loss = evals_result.get("validation_0", {}).get("mlogloss", [])
    test_loss = evals_result.get("validation_1", {}).get("mlogloss", [])
    return [
        {
            "iteration": index + 1,
            "train_mlogloss": float(train_loss[index]) if index < len(train_loss) else None,
            "test_mlogloss": float(value),
        }
        for index, value in enumerate(test_loss)
    ]


def train_model(
    processed_csv: Path,
    model_path: Path,
    metrics_path: Path,
    last_n: int = 5,
    test_fraction: float = 0.2,
) -> dict[str, Any]:
    matches = load_processed_matches(processed_csv)
    features, labels, _ = build_feature_frame(matches, last_n=last_n, include_unfinished=False)
    labels = labels.dropna()
    features = features.loc[labels.index]

    if len(features) < 20:
        raise ValueError(
            f"Need at least 20 finished matches to train; found {len(features)}."
        )
    if labels.nunique() < 2:
        raise ValueError("Need at least two result classes to train a classifier.")

    split_at = int(len(features) * (1.0 - test_fraction))
    split_at = min(max(split_at, 1), len(features) - 1)

    x_train = features.iloc[:split_at]
    y_train = labels.iloc[:split_at]
    x_test = features.iloc[split_at:]
    y_test = labels.iloc[split_at:]

    classifier = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=8,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)
    probabilities = classifier.predict_proba(x_test)
    classes = list(classifier.classes_)
    metrics: dict[str, Any] = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "processed_csv": str(processed_csv),
        "rows": int(len(features)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "last_n": int(last_n),
        "classes": classes,
        "class_meanings": RESULT_LABELS,
        "model_type": "RandomForestClassifier",
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "log_loss": float(log_loss(y_test, probabilities, labels=classes)),
        "classification_report": classification_report(
            y_test, y_pred, labels=classes, output_dict=True, zero_division=0
        ),
    }

    bundle = {
        "model": classifier,
        "model_type": "RandomForestClassifier",
        "feature_columns": list(features.columns),
        "last_n": int(last_n),
        "trained_at": metrics["trained_at"],
        "class_meanings": RESULT_LABELS,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def train_temporal_xgboost(
    processed_csv: Path,
    model_path: Path,
    metrics_path: Path,
    curve_path: Path,
    plot_path: Path | None = None,
    last_n: int = 5,
    train_end_season: int = 2025,
    test_season: int = 2026,
    n_estimators: int = 500,
    learning_rate: float = 0.03,
    max_depth: int = 3,
    use_gpu: bool = True,
) -> dict[str, Any]:
    dataset, feature_columns = _build_training_dataset(processed_csv, last_n)

    train_mask = dataset["season_start_year"].astype(int) <= train_end_season
    test_mask = dataset["season_start_year"].astype(int) == test_season
    if not train_mask.any():
        raise ValueError(f"No training rows found through season {train_end_season}.")
    if not test_mask.any():
        raise ValueError(f"No test rows found for season {test_season}.")

    x_train = dataset.loc[train_mask, feature_columns]
    y_train = dataset.loc[train_mask, "label"]
    x_test = dataset.loc[test_mask, feature_columns]
    y_test = dataset.loc[test_mask, "label"]

    pipeline, label_encoder, device_used, evals_result = _fit_xgboost(
        x_train=x_train,
        y_train=y_train,
        feature_columns=feature_columns,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        use_gpu=use_gpu,
        x_eval=x_test,
        y_eval=y_test,
    )
    evaluation = _evaluate_model(pipeline, label_encoder, x_test, y_test)
    curve_rows = _curve_rows_from_evals(evals_result)
    classifier = pipeline.named_steps["classifier"]
    classes = list(label_encoder.classes_)

    metrics: dict[str, Any] = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "processed_csv": str(processed_csv),
        "model_type": "XGBClassifier",
        "device": device_used,
        "split": {
            "train_end_season": int(train_end_season),
            "test_season": int(test_season),
        },
        "rows": int(len(dataset)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "last_n": int(last_n),
        "classes": classes,
        "class_meanings": RESULT_LABELS,
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "max_depth": int(max_depth),
        "class_weighting": "balanced_sample_weight",
        "accuracy": evaluation["accuracy"],
        "log_loss": evaluation["log_loss"],
        "classification_report": evaluation["classification_report"],
        "top_feature_importances": [
            {"feature": feature, "importance": float(importance)}
            for feature, importance in sorted(
                zip(feature_columns, classifier.feature_importances_),
                key=lambda item: item[1],
                reverse=True,
            )[:25]
        ],
    }

    bundle = {
        "model": pipeline,
        "model_type": "XGBClassifier",
        "label_encoder": label_encoder,
        "feature_columns": feature_columns,
        "last_n": int(last_n),
        "trained_at": metrics["trained_at"],
        "class_meanings": RESULT_LABELS,
        "split": metrics["split"],
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(curve_rows).to_csv(curve_path, index=False)
    if plot_path is not None:
        save_training_curve_plot(curve_path, plot_path)
    return metrics


def backtest_xgboost_grid(
    processed_csv: Path,
    output_path: Path,
    last_n: int = 5,
    n_estimators_values: list[int] | None = None,
    learning_rate_values: list[float] | None = None,
    max_depth_values: list[int] | None = None,
    use_gpu: bool = True,
) -> pd.DataFrame:
    dataset, feature_columns = _build_training_dataset(processed_csv, last_n)
    seasons = sorted(dataset["season_start_year"].astype(int).unique().tolist())
    folds = [(season, season) for season in seasons[1:]]
    if not folds:
        raise ValueError("Need at least two seasons for temporal backtesting.")

    n_estimators_values = n_estimators_values or [50, 75, 100, 150, 200]
    learning_rate_values = learning_rate_values or [0.02, 0.03, 0.05]
    max_depth_values = max_depth_values or [2, 3, 4]

    rows: list[dict[str, Any]] = []
    for n_estimators in n_estimators_values:
        for learning_rate in learning_rate_values:
            for max_depth in max_depth_values:
                fold_metrics = []
                device_used = "cuda" if use_gpu else "cpu"
                for _, validation_season in folds:
                    train_mask = dataset["season_start_year"].astype(int) < validation_season
                    validation_mask = (
                        dataset["season_start_year"].astype(int) == validation_season
                    )
                    if not train_mask.any() or not validation_mask.any():
                        continue

                    x_train = dataset.loc[train_mask, feature_columns]
                    y_train = dataset.loc[train_mask, "label"]
                    x_validation = dataset.loc[validation_mask, feature_columns]
                    y_validation = dataset.loc[validation_mask, "label"]

                    model, label_encoder, device_used, _ = _fit_xgboost(
                        x_train=x_train,
                        y_train=y_train,
                        feature_columns=feature_columns,
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        use_gpu=use_gpu,
                    )
                    evaluation = _evaluate_model(
                        model, label_encoder, x_validation, y_validation
                    )
                    fold_metrics.append(
                        {
                            "validation_season": validation_season,
                            "validation_rows": int(len(x_validation)),
                            "log_loss": evaluation["log_loss"],
                            "accuracy": evaluation["accuracy"],
                        }
                    )

                if not fold_metrics:
                    continue

                rows.append(
                    {
                        "n_estimators": int(n_estimators),
                        "learning_rate": float(learning_rate),
                        "max_depth": int(max_depth),
                        "device": device_used,
                        "folds": len(fold_metrics),
                        "mean_log_loss": float(
                            sum(item["log_loss"] for item in fold_metrics)
                            / len(fold_metrics)
                        ),
                        "mean_accuracy": float(
                            sum(item["accuracy"] for item in fold_metrics)
                            / len(fold_metrics)
                        ),
                        "fold_metrics": json.dumps(fold_metrics),
                    }
                )

    result = pd.DataFrame(rows).sort_values(
        ["mean_log_loss", "mean_accuracy"], ascending=[True, False]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def train_final_xgboost(
    processed_csv: Path,
    model_path: Path,
    metrics_path: Path,
    last_n: int = 5,
    n_estimators: int = 100,
    learning_rate: float = 0.03,
    max_depth: int = 3,
    use_gpu: bool = True,
) -> dict[str, Any]:
    dataset, feature_columns = _build_training_dataset(processed_csv, last_n)
    x_train = dataset[feature_columns]
    y_train = dataset["label"]
    pipeline, label_encoder, device_used, _ = _fit_xgboost(
        x_train=x_train,
        y_train=y_train,
        feature_columns=feature_columns,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        use_gpu=use_gpu,
    )
    classifier = pipeline.named_steps["classifier"]
    classes = list(label_encoder.classes_)
    metrics: dict[str, Any] = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "processed_csv": str(processed_csv),
        "model_type": "XGBClassifier",
        "training_mode": "final_all_finished_matches",
        "device": device_used,
        "rows": int(len(dataset)),
        "last_n": int(last_n),
        "classes": classes,
        "class_meanings": RESULT_LABELS,
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "max_depth": int(max_depth),
        "class_weighting": "balanced_sample_weight",
        "top_feature_importances": [
            {"feature": feature, "importance": float(importance)}
            for feature, importance in sorted(
                zip(feature_columns, classifier.feature_importances_),
                key=lambda item: item[1],
                reverse=True,
            )[:25]
        ],
    }
    bundle = {
        "model": pipeline,
        "model_type": "XGBClassifier",
        "label_encoder": label_encoder,
        "feature_columns": feature_columns,
        "last_n": int(last_n),
        "trained_at": metrics["trained_at"],
        "class_meanings": RESULT_LABELS,
        "training_mode": metrics["training_mode"],
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_model(model_path: Path) -> dict[str, Any]:
    return joblib.load(model_path)


def predict_features(bundle: dict[str, Any], features: pd.DataFrame) -> pd.DataFrame:
    feature_columns = bundle["feature_columns"]
    missing = sorted(set(feature_columns) - set(features.columns))
    if missing:
        raise ValueError(f"Prediction features missing columns: {missing}")

    model = bundle["model"]
    aligned = features[feature_columns]
    probabilities = model.predict_proba(aligned)
    label_encoder = bundle.get("label_encoder")
    if label_encoder is not None:
        classes = list(label_encoder.classes_)
        raw_prediction = model.predict(aligned).astype(int)
        prediction = label_encoder.inverse_transform(raw_prediction)
    else:
        classes = list(model.classes_)
        prediction = model.predict(aligned)
    output = pd.DataFrame(probabilities, columns=[f"prob_{klass}" for klass in classes])
    output["prediction"] = prediction
    output["prediction_label"] = output["prediction"].map(RESULT_LABELS)
    return output
