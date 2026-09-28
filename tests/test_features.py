from __future__ import annotations

import pandas as pd

from match_predictor.features import build_feature_frame


def test_features_use_only_prior_matches() -> None:
    matches = pd.DataFrame(
        [
            {
                "match_id": 1,
                "utc_date": pd.Timestamp("2024-08-01T12:00:00Z"),
                "status": "FINISHED",
                "home_team_id": 1,
                "home_team": "Alpha FC",
                "away_team_id": 2,
                "away_team": "Beta FC",
                "home_goals": 2,
                "away_goals": 0,
                "outcome": "H",
            },
            {
                "match_id": 2,
                "utc_date": pd.Timestamp("2024-08-08T12:00:00Z"),
                "status": "FINISHED",
                "home_team_id": 2,
                "home_team": "Beta FC",
                "away_team_id": 1,
                "away_team": "Alpha FC",
                "home_goals": 1,
                "away_goals": 1,
                "outcome": "D",
            },
        ]
    )

    features, labels, meta = build_feature_frame(matches, last_n=5)

    assert labels.tolist() == ["H", "D"]
    assert meta["match_id"].tolist() == [1, 2]
    assert features.iloc[0]["home_last5_points"] == 1.0
    assert features.iloc[1]["home_last5_points"] == 0.0
    assert features.iloc[1]["away_last5_points"] == 3.0


def test_head_to_head_features_are_based_on_previous_meetings() -> None:
    matches = pd.DataFrame(
        [
            {
                "match_id": 1,
                "utc_date": pd.Timestamp("2024-08-01T12:00:00Z"),
                "status": "FINISHED",
                "home_team_id": 1,
                "home_team": "Alpha FC",
                "away_team_id": 2,
                "away_team": "Beta FC",
                "home_goals": 3,
                "away_goals": 1,
                "outcome": "H",
            },
            {
                "match_id": 2,
                "utc_date": pd.Timestamp("2024-09-01T12:00:00Z"),
                "status": "SCHEDULED",
                "home_team_id": 2,
                "home_team": "Beta FC",
                "away_team_id": 1,
                "away_team": "Alpha FC",
                "home_goals": None,
                "away_goals": None,
                "outcome": None,
            },
        ]
    )

    features, labels, meta = build_feature_frame(matches, last_n=5, include_unfinished=True)

    scheduled_index = meta.index[meta["match_id"] == 2][0]
    assert pd.isna(labels.iloc[scheduled_index])
    assert features.iloc[scheduled_index]["h2h_last5_matches"] == 1.0
    assert features.iloc[scheduled_index]["h2h_last5_home_points"] == 0.0
    assert features.iloc[scheduled_index]["h2h_last5_away_points"] == 3.0


def test_historical_and_venue_features_use_previous_matches_only() -> None:
    matches = pd.DataFrame(
        [
            {
                "match_id": 1,
                "utc_date": pd.Timestamp("2024-08-01T12:00:00Z"),
                "status": "FINISHED",
                "home_team_id": 1,
                "home_team": "Alpha FC",
                "away_team_id": 2,
                "away_team": "Beta FC",
                "home_goals": 2,
                "away_goals": 0,
                "outcome": "H",
            },
            {
                "match_id": 2,
                "utc_date": pd.Timestamp("2024-08-08T12:00:00Z"),
                "status": "FINISHED",
                "home_team_id": 1,
                "home_team": "Alpha FC",
                "away_team_id": 3,
                "away_team": "Gamma FC",
                "home_goals": 1,
                "away_goals": 1,
                "outcome": "D",
            },
            {
                "match_id": 3,
                "utc_date": pd.Timestamp("2024-08-15T12:00:00Z"),
                "status": "SCHEDULED",
                "home_team_id": 1,
                "home_team": "Alpha FC",
                "away_team_id": 2,
                "away_team": "Beta FC",
                "home_goals": None,
                "away_goals": None,
                "outcome": None,
            },
        ]
    )

    features, _, meta = build_feature_frame(matches, last_n=5, include_unfinished=True)

    target_index = meta.index[meta["match_id"] == 3][0]
    target = features.iloc[target_index]
    assert target["home_played"] == 2.0
    assert target["home_all_goals_for"] == 1.5
    assert target["home_all_goals_against"] == 0.5
    assert target["home_at_home_played"] == 2.0
    assert target["home_at_home_all_goals_for"] == 1.5
    assert target["h2h_all_matches"] == 1.0
