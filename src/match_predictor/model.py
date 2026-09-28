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

    train_mask = dataset["season_start_year"].astype(int) <= train_end_season
    test_mask = dataset["season_start_year"].astype(int) == test_season
    if not train_mask.any():
        raise ValueError(f"No training rows found through season {train_end_season}.")
    if not test_mask.any():
        raise ValueError(f"No test rows found for season {test_season}.")

    feature_columns = list(features.columns)
    x_train = dataset.loc[train_mask, feature_columns]
    y_train = dataset.loc[train_mask, "label"]
    x_test = dataset.loc[test_mask, feature_columns]
    y_test = dataset.loc[test_mask, "label"]

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train_encoded)
    classes = list(label_encoder.classes_)
    if len(classes) < 2:
        raise ValueError("Need at least two result classes to train a classifier.")

    imputer = SimpleImputer(strategy="median")
    x_train_imputed = imputer.fit_transform(x_train)
    x_test_imputed = imputer.transform(x_test)

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
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
    )

    try:
        classifier.fit(
            x_train_imputed,
            y_train_encoded,
            sample_weight=sample_weight,
            eval_set=[(x_train_imputed, y_train_encoded), (x_test_imputed, y_test_encoded)],
            verbose=False,
        )
    except Exception:
        if not use_gpu:
            raise
        device_used = "cpu"
        classifier.set_params(device="cpu")
        classifier.fit(
            x_train_imputed,
            y_train_encoded,
            sample_weight=sample_weight,
            eval_set=[(x_train_imputed, y_train_encoded), (x_test_imputed, y_test_encoded)],
            verbose=False,
        )

    probabilities = classifier.predict_proba(x_test_imputed)
    predictions_encoded = classifier.predict(x_test_imputed).astype(int)
    predictions = label_encoder.inverse_transform(predictions_encoded)
    evals_result = classifier.evals_result()
    train_loss = evals_result.get("validation_0", {}).get("mlogloss", [])
    test_loss = evals_result.get("validation_1", {}).get("mlogloss", [])
    curve_rows = [
        {
            "iteration": index + 1,
            "train_mlogloss": float(train_loss[index]) if index < len(train_loss) else None,
            "test_mlogloss": float(value),
        }
        for index, value in enumerate(test_loss)
    ]

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
        "accuracy": float(accuracy_score(y_test, predictions)),
        "log_loss": float(log_loss(y_test, probabilities, labels=classes)),
        "classification_report": classification_report(
            y_test, predictions, labels=classes, output_dict=True, zero_division=0
        ),
        "top_feature_importances": [
            {"feature": feature, "importance": float(importance)}
            for feature, importance in sorted(
                zip(feature_columns, classifier.feature_importances_),
                key=lambda item: item[1],
                reverse=True,
            )[:25]
        ],
    }

    pipeline = Pipeline(
        steps=[
            ("imputer", imputer),
            ("classifier", classifier),
        ]
    )
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
