from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .api import FootballDataClient, fetch_competition_matches
from .data import load_processed_matches, rebuild_processed_csv
from .features import build_feature_frame
from .model import (
    backtest_xgboost_grid,
    default_curve_plot_path,
    default_curve_path,
    default_metrics_path,
    default_model_path,
    load_model,
    predict_features,
    train_model,
    train_final_xgboost,
    train_temporal_xgboost,
)


DATA_DIR = Path("data")
MODELS_DIR = Path("models")
DEFAULT_COMPETITION = "PPL"


def _processed_path(data_dir: Path, competition: str) -> Path:
    return data_dir / "processed" / f"{competition.upper()}_matches.csv"


def _fetch(args: argparse.Namespace) -> None:
    client = FootballDataClient(api_key=args.api_key)
    seasons = args.seasons or [None]
    for season in seasons:
        path = fetch_competition_matches(
            client=client,
            competition=args.competition,
            season=season,
            status=args.status,
            cache_dir=args.data_dir / "raw",
            force=args.force,
        )
        print(f"cached {path}")

    processed = rebuild_processed_csv(args.data_dir, args.competition)
    print(f"rebuilt {processed}")


def _train(args: argparse.Namespace) -> None:
    processed_csv = args.processed_csv or _processed_path(args.data_dir, args.competition)
    model_path = args.model_path or default_model_path(args.models_dir, args.competition)
    metrics_path = args.metrics_path or default_metrics_path(args.models_dir, args.competition)
    metrics = train_model(
        processed_csv=processed_csv,
        model_path=model_path,
        metrics_path=metrics_path,
        last_n=args.last_n,
        test_fraction=args.test_fraction,
    )
    print(f"saved model {model_path}")
    print(f"saved metrics {metrics_path}")
    print(
        f"accuracy={metrics['accuracy']:.3f} log_loss={metrics['log_loss']:.3f} "
        f"rows={metrics['rows']} test_rows={metrics['test_rows']}"
    )


def _train_temporal(args: argparse.Namespace) -> None:
    processed_csv = args.processed_csv or _processed_path(args.data_dir, args.competition)
    model_path = args.model_path or default_model_path(args.models_dir, args.competition)
    metrics_path = args.metrics_path or default_metrics_path(args.models_dir, args.competition)
    curve_path = args.curve_path or default_curve_path(args.models_dir, args.competition)
    curve_plot_path = args.curve_plot_path or default_curve_plot_path(
        args.models_dir, args.competition
    )
    metrics = train_temporal_xgboost(
        processed_csv=processed_csv,
        model_path=model_path,
        metrics_path=metrics_path,
        curve_path=curve_path,
        plot_path=curve_plot_path,
        last_n=args.last_n,
        train_end_season=args.train_end_season,
        test_season=args.test_season,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        use_gpu=not args.no_gpu,
    )
    curve = pd.read_csv(curve_path)
    print(curve.to_string(index=False))
    print(f"saved model {model_path}")
    print(f"saved metrics {metrics_path}")
    print(f"saved training curve {curve_path}")
    print(f"saved training curve plot {curve_plot_path}")
    print(
        f"final accuracy={metrics['accuracy']:.3f} log_loss={metrics['log_loss']:.3f} "
        f"device={metrics['device']} train_rows={metrics['train_rows']} "
        f"test_rows={metrics['test_rows']}"
    )


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _backtest_xgboost(args: argparse.Namespace) -> None:
    processed_csv = args.processed_csv or _processed_path(args.data_dir, args.competition)
    output_path = args.output_path or (
        args.models_dir / f"{args.competition.upper()}_backtest_grid.csv"
    )
    result = backtest_xgboost_grid(
        processed_csv=processed_csv,
        output_path=output_path,
        last_n=args.last_n,
        n_estimators_values=_parse_int_list(args.n_estimators),
        learning_rate_values=_parse_float_list(args.learning_rates),
        max_depth_values=_parse_int_list(args.max_depths),
        use_gpu=not args.no_gpu,
    )
    columns = [
        "n_estimators",
        "learning_rate",
        "max_depth",
        "folds",
        "mean_log_loss",
        "mean_accuracy",
        "device",
    ]
    print(result[columns].head(args.top).to_string(index=False))
    print(f"saved backtest grid {output_path}")


def _train_final(args: argparse.Namespace) -> None:
    processed_csv = args.processed_csv or _processed_path(args.data_dir, args.competition)
    model_path = args.model_path or default_model_path(args.models_dir, args.competition)
    metrics_path = args.metrics_path or default_metrics_path(args.models_dir, args.competition)
    metrics = train_final_xgboost(
        processed_csv=processed_csv,
        model_path=model_path,
        metrics_path=metrics_path,
        last_n=args.last_n,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        use_gpu=not args.no_gpu,
    )
    print(f"saved final model {model_path}")
    print(f"saved final metrics {metrics_path}")
    print(
        f"rows={metrics['rows']} n_estimators={metrics['n_estimators']} "
        f"learning_rate={metrics['learning_rate']} max_depth={metrics['max_depth']} "
        f"device={metrics['device']}"
    )


def _prediction_table(result: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "utc_date",
        "home_team",
        "away_team",
        "prob_H",
        "prob_D",
        "prob_A",
        "prediction_label",
    ]
    existing_columns = [column for column in columns if column in result.columns]
    table = result[existing_columns].copy()
    for column in ("prob_H", "prob_D", "prob_A"):
        if column in table:
            table[column] = (table[column] * 100).round(1)
    return table


def _predict_upcoming(args: argparse.Namespace) -> None:
    if args.fetch:
        client = FootballDataClient(api_key=args.api_key)
        fetch_competition_matches(
            client=client,
            competition=args.competition,
            season=args.season,
            status="SCHEDULED",
            cache_dir=args.data_dir / "raw",
            force=True,
        )
        rebuild_processed_csv(args.data_dir, args.competition)

    processed_csv = args.processed_csv or _processed_path(args.data_dir, args.competition)
    model_path = args.model_path or default_model_path(args.models_dir, args.competition)
    matches = load_processed_matches(processed_csv)
    bundle = load_model(model_path)
    features, labels, meta = build_feature_frame(
        matches, last_n=bundle["last_n"], include_unfinished=True
    )
    future_mask = labels.isna() & meta["status"].isin(["SCHEDULED", "TIMED", "POSTPONED"])
    prediction_features = features.loc[future_mask].head(args.limit).reset_index(drop=True)
    prediction_meta = meta.loc[future_mask].head(args.limit).reset_index(drop=True)

    if prediction_features.empty:
        print("No upcoming matches found in the processed data.")
        return

    predictions = predict_features(bundle, prediction_features)
    result = pd.concat([prediction_meta, predictions], axis=1)
    output_dir = args.data_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.competition.upper()}_upcoming_predictions.csv"
    result.to_csv(output_path, index=False)

    print(_prediction_table(result).to_string(index=False))
    print(f"saved predictions {output_path}")


def _find_team(matches: pd.DataFrame, query: str) -> tuple[float, str]:
    teams = pd.concat(
        [
            matches[["home_team_id", "home_team"]].rename(
                columns={"home_team_id": "team_id", "home_team": "team"}
            ),
            matches[["away_team_id", "away_team"]].rename(
                columns={"away_team_id": "team_id", "away_team": "team"}
            ),
        ],
        ignore_index=True,
    ).dropna(subset=["team"])
    normalized = query.strip().lower()
    exact = teams[teams["team"].str.lower() == normalized]
    if exact.empty:
        exact = teams[teams["team"].str.lower().str.contains(normalized, regex=False)]
    if exact.empty:
        raise ValueError(f"Could not find a team matching {query!r} in cached data.")
    row = exact.drop_duplicates("team_id").iloc[0]
    return row["team_id"], row["team"]


def _predict_match(args: argparse.Namespace) -> None:
    processed_csv = args.processed_csv or _processed_path(args.data_dir, args.competition)
    model_path = args.model_path or default_model_path(args.models_dir, args.competition)
    matches = load_processed_matches(processed_csv)
    bundle = load_model(model_path)

    home_id, home_name = _find_team(matches, args.home)
    away_id, away_name = _find_team(matches, args.away)
    manual_match = {
        "match_id": -1,
        "competition": args.competition.upper(),
        "season_start_year": None,
        "utc_date": pd.Timestamp(args.utc_date) if args.utc_date else datetime.now(timezone.utc),
        "status": "SCHEDULED",
        "matchday": None,
        "home_team_id": home_id,
        "home_team": home_name,
        "away_team_id": away_id,
        "away_team": away_name,
        "home_goals": None,
        "away_goals": None,
        "outcome": None,
    }
    augmented = pd.concat([matches, pd.DataFrame([manual_match])], ignore_index=True)
    features, labels, meta = build_feature_frame(
        augmented, last_n=bundle["last_n"], include_unfinished=True
    )
    target_index = meta.index[meta["match_id"] == -1]
    if target_index.empty:
        raise ValueError("Could not build features for the manual match.")

    prediction = predict_features(bundle, features.loc[target_index].reset_index(drop=True))
    result = pd.concat([meta.loc[target_index].reset_index(drop=True), prediction], axis=1)
    print(_prediction_table(result).to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Football match predictor")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--api-key", default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser("fetch", help="Fetch and cache match data")
    fetch.add_argument("--competition", default=DEFAULT_COMPETITION)
    fetch.add_argument("--seasons", nargs="*", type=int)
    fetch.add_argument("--status", default="FINISHED")
    fetch.add_argument("--force", action="store_true")
    fetch.set_defaults(func=_fetch)

    train = subparsers.add_parser("train", help="Train a prediction model")
    train.add_argument("--competition", default=DEFAULT_COMPETITION)
    train.add_argument("--processed-csv", type=Path)
    train.add_argument("--model-path", type=Path)
    train.add_argument("--metrics-path", type=Path)
    train.add_argument("--last-n", type=int, default=5)
    train.add_argument("--test-fraction", type=float, default=0.2)
    train.set_defaults(func=_train)

    temporal = subparsers.add_parser(
        "train-temporal",
        help="Train XGBoost with a season-based split and mlogloss curve",
    )
    temporal.add_argument("--competition", default=DEFAULT_COMPETITION)
    temporal.add_argument("--processed-csv", type=Path)
    temporal.add_argument("--model-path", type=Path)
    temporal.add_argument("--metrics-path", type=Path)
    temporal.add_argument("--curve-path", type=Path)
    temporal.add_argument("--curve-plot-path", type=Path)
    temporal.add_argument("--last-n", type=int, default=5)
    temporal.add_argument("--train-end-season", type=int, default=2025)
    temporal.add_argument("--test-season", type=int, default=2026)
    temporal.add_argument("--n-estimators", type=int, default=500)
    temporal.add_argument("--learning-rate", type=float, default=0.03)
    temporal.add_argument("--max-depth", type=int, default=3)
    temporal.add_argument("--no-gpu", action="store_true")
    temporal.set_defaults(func=_train_temporal)

    backtest = subparsers.add_parser(
        "backtest-xgboost",
        help="Run temporal grid backtests to choose XGBoost hyperparameters",
    )
    backtest.add_argument("--competition", default=DEFAULT_COMPETITION)
    backtest.add_argument("--processed-csv", type=Path)
    backtest.add_argument("--output-path", type=Path)
    backtest.add_argument("--last-n", type=int, default=5)
    backtest.add_argument("--n-estimators", default="50,75,100,150,200")
    backtest.add_argument("--learning-rates", default="0.02,0.03,0.05")
    backtest.add_argument("--max-depths", default="2,3,4")
    backtest.add_argument("--top", type=int, default=10)
    backtest.add_argument("--no-gpu", action="store_true")
    backtest.set_defaults(func=_backtest_xgboost)

    final = subparsers.add_parser(
        "train-final",
        help="Train final XGBoost on all finished matches with chosen hyperparameters",
    )
    final.add_argument("--competition", default=DEFAULT_COMPETITION)
    final.add_argument("--processed-csv", type=Path)
    final.add_argument("--model-path", type=Path)
    final.add_argument("--metrics-path", type=Path)
    final.add_argument("--last-n", type=int, default=5)
    final.add_argument("--n-estimators", type=int, default=100)
    final.add_argument("--learning-rate", type=float, default=0.03)
    final.add_argument("--max-depth", type=int, default=3)
    final.add_argument("--no-gpu", action="store_true")
    final.set_defaults(func=_train_final)

    upcoming = subparsers.add_parser("predict-upcoming", help="Predict scheduled matches")
    upcoming.add_argument("--competition", default=DEFAULT_COMPETITION)
    upcoming.add_argument("--season", type=int)
    upcoming.add_argument("--processed-csv", type=Path)
    upcoming.add_argument("--model-path", type=Path)
    upcoming.add_argument("--limit", type=int, default=20)
    upcoming.add_argument("--no-fetch", dest="fetch", action="store_false")
    upcoming.set_defaults(func=_predict_upcoming, fetch=True)

    manual = subparsers.add_parser("predict-match", help="Predict one manual fixture")
    manual.add_argument("--competition", default=DEFAULT_COMPETITION)
    manual.add_argument("--home", required=True)
    manual.add_argument("--away", required=True)
    manual.add_argument("--utc-date")
    manual.add_argument("--processed-csv", type=Path)
    manual.add_argument("--model-path", type=Path)
    manual.set_defaults(func=_predict_match)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
