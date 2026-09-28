from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import log
from typing import Any

import pandas as pd


RESULT_LABELS = {"H": "home_win", "D": "draw", "A": "away_win"}


@dataclass
class VenueState:
    played: int = 0
    points_total: float = 0.0
    goals_for_total: float = 0.0
    goals_against_total: float = 0.0
    wins_total: float = 0.0
    draws_total: float = 0.0
    losses_total: float = 0.0
    points: deque[float] = field(default_factory=deque)
    goals_for: deque[float] = field(default_factory=deque)
    goals_against: deque[float] = field(default_factory=deque)
    goal_diff: deque[float] = field(default_factory=deque)


@dataclass
class TeamState:
    elo: float = 1500.0
    played: int = 0
    points_total: float = 0.0
    goals_for_total: float = 0.0
    goals_against_total: float = 0.0
    wins_total: float = 0.0
    draws_total: float = 0.0
    losses_total: float = 0.0
    last_date: pd.Timestamp | None = None
    points: deque[float] = field(default_factory=deque)
    goals_for: deque[float] = field(default_factory=deque)
    goals_against: deque[float] = field(default_factory=deque)
    goal_diff: deque[float] = field(default_factory=deque)
    wins: deque[float] = field(default_factory=deque)
    draws: deque[float] = field(default_factory=deque)
    losses: deque[float] = field(default_factory=deque)
    home: VenueState = field(default_factory=VenueState)
    away: VenueState = field(default_factory=VenueState)


@dataclass
class HeadToHeadState:
    played: int = 0
    home_points_total: float = 0.0
    away_points_total: float = 0.0
    home_goals_total: float = 0.0
    away_goals_total: float = 0.0
    draws_total: float = 0.0
    home_points: deque[float] = field(default_factory=deque)
    away_points: deque[float] = field(default_factory=deque)
    home_goals: deque[float] = field(default_factory=deque)
    away_goals: deque[float] = field(default_factory=deque)
    draws: deque[float] = field(default_factory=deque)


@dataclass
class RefereeState:
    matches: int = 0
    home_wins: float = 0.0
    draws: float = 0.0
    away_wins: float = 0.0
    home_goals_total: float = 0.0
    away_goals_total: float = 0.0


def _team_key(row: pd.Series, side: str) -> str:
    team_id = row.get(f"{side}_team_id")
    if pd.notna(team_id):
        return f"id:{int(team_id)}"
    return f"name:{str(row.get(f'{side}_team')).strip().lower()}"


def _mean(values: deque[float], default: float) -> float:
    return float(sum(values) / len(values)) if values else default


def _rate(numerator: float, denominator: int, default: float) -> float:
    return float(numerator / denominator) if denominator else default


def _rest_days(state: TeamState, match_date: pd.Timestamp | None) -> float:
    if state.last_date is None or match_date is None or pd.isna(match_date):
        return 7.0
    return max(0.0, min(30.0, (match_date - state.last_date).total_seconds() / 86400.0))


def _features_for_team(
    state: TeamState, prefix: str, match_date: pd.Timestamp | None, last_n: int
) -> dict[str, float]:
    venue = state.home if prefix == "home" else state.away
    venue_label = "at_home" if prefix == "home" else "at_away"
    return {
        f"{prefix}_elo": state.elo,
        f"{prefix}_played": float(state.played),
        f"{prefix}_all_points_per_game": _rate(state.points_total, state.played, 1.0),
        f"{prefix}_all_goals_for": _rate(state.goals_for_total, state.played, 1.2),
        f"{prefix}_all_goals_against": _rate(
            state.goals_against_total, state.played, 1.2
        ),
        f"{prefix}_all_goal_diff": _rate(
            state.goals_for_total - state.goals_against_total, state.played, 0.0
        ),
        f"{prefix}_all_win_rate": _rate(state.wins_total, state.played, 0.33),
        f"{prefix}_all_draw_rate": _rate(state.draws_total, state.played, 0.27),
        f"{prefix}_all_loss_rate": _rate(state.losses_total, state.played, 0.40),
        f"{prefix}_last{last_n}_points": _mean(state.points, 1.0),
        f"{prefix}_last{last_n}_goals_for": _mean(state.goals_for, 1.2),
        f"{prefix}_last{last_n}_goals_against": _mean(state.goals_against, 1.2),
        f"{prefix}_last{last_n}_goal_diff": _mean(state.goal_diff, 0.0),
        f"{prefix}_last{last_n}_win_rate": _mean(state.wins, 0.33),
        f"{prefix}_last{last_n}_draw_rate": _mean(state.draws, 0.27),
        f"{prefix}_last{last_n}_loss_rate": _mean(state.losses, 0.40),
        f"{prefix}_rest_days": _rest_days(state, match_date),
        f"{prefix}_{venue_label}_played": float(venue.played),
        f"{prefix}_{venue_label}_all_points_per_game": _rate(
            venue.points_total, venue.played, 1.0
        ),
        f"{prefix}_{venue_label}_all_goals_for": _rate(
            venue.goals_for_total, venue.played, 1.2
        ),
        f"{prefix}_{venue_label}_all_goals_against": _rate(
            venue.goals_against_total, venue.played, 1.2
        ),
        f"{prefix}_{venue_label}_all_goal_diff": _rate(
            venue.goals_for_total - venue.goals_against_total, venue.played, 0.0
        ),
        f"{prefix}_{venue_label}_all_win_rate": _rate(
            venue.wins_total, venue.played, 0.33
        ),
        f"{prefix}_{venue_label}_all_draw_rate": _rate(
            venue.draws_total, venue.played, 0.27
        ),
        f"{prefix}_{venue_label}_all_loss_rate": _rate(
            venue.losses_total, venue.played, 0.40
        ),
        f"{prefix}_{venue_label}_last{last_n}_points": _mean(venue.points, 1.0),
        f"{prefix}_{venue_label}_last{last_n}_goals_for": _mean(
            venue.goals_for, 1.2
        ),
        f"{prefix}_{venue_label}_last{last_n}_goals_against": _mean(
            venue.goals_against, 1.2
        ),
        f"{prefix}_{venue_label}_last{last_n}_goal_diff": _mean(
            venue.goal_diff, 0.0
        ),
    }


def _trim(state: TeamState, last_n: int) -> None:
    for values in (
        state.points,
        state.goals_for,
        state.goals_against,
        state.goal_diff,
        state.wins,
        state.draws,
        state.losses,
    ):
        while len(values) > last_n:
            values.popleft()


def _trim_venue(state: VenueState, last_n: int) -> None:
    for values in (
        state.points,
        state.goals_for,
        state.goals_against,
        state.goal_diff,
    ):
        while len(values) > last_n:
            values.popleft()


def _trim_h2h(state: HeadToHeadState, last_n: int) -> None:
    for values in (
        state.home_points,
        state.away_points,
        state.home_goals,
        state.away_goals,
        state.draws,
    ):
        while len(values) > last_n:
            values.popleft()


def _points_for(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def _pair_key(home_key: str, away_key: str) -> tuple[str, str]:
    return tuple(sorted((home_key, away_key)))


def _h2h_features(
    h2h: HeadToHeadState,
    home_key: str,
    away_key: str,
    canonical_home_key: str,
    last_n: int,
) -> dict[str, float]:
    home_is_canonical = home_key == canonical_home_key
    canonical_home_points = _mean(h2h.home_points, 1.0)
    canonical_away_points = _mean(h2h.away_points, 1.0)
    canonical_home_goals = _mean(h2h.home_goals, 1.2)
    canonical_away_goals = _mean(h2h.away_goals, 1.2)
    canonical_home_all_points = _rate(h2h.home_points_total, h2h.played, 1.0)
    canonical_away_all_points = _rate(h2h.away_points_total, h2h.played, 1.0)
    canonical_home_all_goals = _rate(h2h.home_goals_total, h2h.played, 1.2)
    canonical_away_all_goals = _rate(h2h.away_goals_total, h2h.played, 1.2)

    if home_is_canonical:
        fixture_home_points = canonical_home_points
        fixture_away_points = canonical_away_points
        fixture_home_goals = canonical_home_goals
        fixture_away_goals = canonical_away_goals
        fixture_home_all_points = canonical_home_all_points
        fixture_away_all_points = canonical_away_all_points
        fixture_home_all_goals = canonical_home_all_goals
        fixture_away_all_goals = canonical_away_all_goals
    else:
        fixture_home_points = canonical_away_points
        fixture_away_points = canonical_home_points
        fixture_home_goals = canonical_away_goals
        fixture_away_goals = canonical_home_goals
        fixture_home_all_points = canonical_away_all_points
        fixture_away_all_points = canonical_home_all_points
        fixture_home_all_goals = canonical_away_all_goals
        fixture_away_all_goals = canonical_home_all_goals

    return {
        "h2h_all_matches": float(h2h.played),
        "h2h_all_home_points": fixture_home_all_points,
        "h2h_all_away_points": fixture_away_all_points,
        "h2h_all_points_diff": fixture_home_all_points - fixture_away_all_points,
        "h2h_all_home_goals": fixture_home_all_goals,
        "h2h_all_away_goals": fixture_away_all_goals,
        "h2h_all_goal_diff": fixture_home_all_goals - fixture_away_all_goals,
        "h2h_all_draw_rate": _rate(h2h.draws_total, h2h.played, 0.27),
        f"h2h_last{last_n}_home_points": fixture_home_points,
        f"h2h_last{last_n}_away_points": fixture_away_points,
        f"h2h_last{last_n}_points_diff": fixture_home_points - fixture_away_points,
        f"h2h_last{last_n}_home_goals": fixture_home_goals,
        f"h2h_last{last_n}_away_goals": fixture_away_goals,
        f"h2h_last{last_n}_goal_diff": fixture_home_goals - fixture_away_goals,
        f"h2h_last{last_n}_draw_rate": _mean(h2h.draws, 0.27),
        f"h2h_last{last_n}_matches": float(len(h2h.draws)),
    }


def _update_h2h(
    state: HeadToHeadState,
    home_key: str,
    away_key: str,
    canonical_home_key: str,
    home_goals: int,
    away_goals: int,
    last_n: int,
) -> None:
    home_points = _points_for(home_goals, away_goals)
    away_points = _points_for(away_goals, home_goals)
    state.played += 1
    state.draws_total += 1.0 if home_goals == away_goals else 0.0

    if home_key == canonical_home_key:
        state.home_points_total += float(home_points)
        state.away_points_total += float(away_points)
        state.home_goals_total += float(home_goals)
        state.away_goals_total += float(away_goals)
        state.home_points.append(float(home_points))
        state.away_points.append(float(away_points))
        state.home_goals.append(float(home_goals))
        state.away_goals.append(float(away_goals))
    else:
        state.home_points_total += float(away_points)
        state.away_points_total += float(home_points)
        state.home_goals_total += float(away_goals)
        state.away_goals_total += float(home_goals)
        state.home_points.append(float(away_points))
        state.away_points.append(float(home_points))
        state.home_goals.append(float(away_goals))
        state.away_goals.append(float(home_goals))

    state.draws.append(1.0 if home_goals == away_goals else 0.0)
    _trim_h2h(state, last_n)


def _update_elo(home: TeamState, away: TeamState, home_goals: int, away_goals: int) -> None:
    home_advantage = 65.0
    expected_home = 1.0 / (1.0 + 10.0 ** ((away.elo - (home.elo + home_advantage)) / 400.0))
    actual_home = 1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0
    margin = abs(home_goals - away_goals)
    margin_multiplier = 1.0 if margin <= 1 else log(margin + 1.0)
    k_factor = 20.0

    change = k_factor * margin_multiplier * (actual_home - expected_home)
    home.elo += change
    away.elo -= change


def _update_team(
    state: TeamState,
    match_date: pd.Timestamp,
    goals_for: int,
    goals_against: int,
    last_n: int,
    venue: str,
) -> None:
    points = _points_for(goals_for, goals_against)
    state.played += 1
    state.points_total += float(points)
    state.goals_for_total += float(goals_for)
    state.goals_against_total += float(goals_against)
    state.wins_total += 1.0 if points == 3 else 0.0
    state.draws_total += 1.0 if points == 1 else 0.0
    state.losses_total += 1.0 if points == 0 else 0.0
    state.last_date = match_date
    state.points.append(float(points))
    state.goals_for.append(float(goals_for))
    state.goals_against.append(float(goals_against))
    state.goal_diff.append(float(goals_for - goals_against))
    state.wins.append(1.0 if points == 3 else 0.0)
    state.draws.append(1.0 if points == 1 else 0.0)
    state.losses.append(1.0 if points == 0 else 0.0)
    _trim(state, last_n)

    venue_state = state.home if venue == "home" else state.away
    venue_state.played += 1
    venue_state.points_total += float(points)
    venue_state.goals_for_total += float(goals_for)
    venue_state.goals_against_total += float(goals_against)
    venue_state.wins_total += 1.0 if points == 3 else 0.0
    venue_state.draws_total += 1.0 if points == 1 else 0.0
    venue_state.losses_total += 1.0 if points == 0 else 0.0
    venue_state.points.append(float(points))
    venue_state.goals_for.append(float(goals_for))
    venue_state.goals_against.append(float(goals_against))
    venue_state.goal_diff.append(float(goals_for - goals_against))
    _trim_venue(venue_state, last_n)


def _finished_with_score(row: pd.Series) -> bool:
    return (
        row.get("status") == "FINISHED"
        and pd.notna(row.get("home_goals"))
        and pd.notna(row.get("away_goals"))
        and row.get("outcome") in RESULT_LABELS
    )


def _date_features(row: pd.Series) -> dict[str, float]:
    match_date = row.get("utc_date")
    if match_date is None or pd.isna(match_date):
        return {
            "matchday_number": float(row.get("matchday") or 0.0),
            "kickoff_hour_utc": 19.0,
            "kickoff_dayofweek": 5.0,
            "kickoff_month": 8.0,
            "is_weekend": 1.0,
            "season_progress": 0.5,
        }

    matchday = row.get("matchday")
    matchday_value = float(matchday) if pd.notna(matchday) else 0.0
    dayofweek = float(match_date.dayofweek)
    return {
        "matchday_number": matchday_value,
        "kickoff_hour_utc": float(match_date.hour + match_date.minute / 60.0),
        "kickoff_dayofweek": dayofweek,
        "kickoff_month": float(match_date.month),
        "is_weekend": 1.0 if dayofweek >= 5.0 else 0.0,
        "season_progress": max(0.0, min(1.0, matchday_value / 34.0))
        if matchday_value
        else 0.5,
    }


def _referee_key(row: pd.Series) -> str | None:
    referee_id = row.get("referee_id")
    if pd.notna(referee_id):
        return f"id:{int(referee_id)}"
    referee_name = row.get("referee_name")
    if pd.notna(referee_name) and str(referee_name).strip():
        return f"name:{str(referee_name).strip().lower()}"
    return None


def _referee_features(referee: RefereeState | None) -> dict[str, float]:
    if referee is None:
        return {
            "has_referee": 0.0,
            "referee_prior_matches": 0.0,
            "referee_prior_home_win_rate": 0.33,
            "referee_prior_draw_rate": 0.27,
            "referee_prior_away_win_rate": 0.40,
            "referee_prior_home_goals": 1.3,
            "referee_prior_away_goals": 1.1,
            "referee_prior_total_goals": 2.4,
        }

    return {
        "has_referee": 1.0,
        "referee_prior_matches": float(referee.matches),
        "referee_prior_home_win_rate": _rate(referee.home_wins, referee.matches, 0.33),
        "referee_prior_draw_rate": _rate(referee.draws, referee.matches, 0.27),
        "referee_prior_away_win_rate": _rate(referee.away_wins, referee.matches, 0.40),
        "referee_prior_home_goals": _rate(
            referee.home_goals_total, referee.matches, 1.3
        ),
        "referee_prior_away_goals": _rate(
            referee.away_goals_total, referee.matches, 1.1
        ),
        "referee_prior_total_goals": _rate(
            referee.home_goals_total + referee.away_goals_total, referee.matches, 2.4
        ),
    }


def _update_referee(
    referee: RefereeState, home_goals: int, away_goals: int, outcome: str
) -> None:
    referee.matches += 1
    referee.home_wins += 1.0 if outcome == "H" else 0.0
    referee.draws += 1.0 if outcome == "D" else 0.0
    referee.away_wins += 1.0 if outcome == "A" else 0.0
    referee.home_goals_total += float(home_goals)
    referee.away_goals_total += float(away_goals)


def build_feature_frame(
    matches: pd.DataFrame,
    last_n: int = 5,
    include_unfinished: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build pre-match features without leaking the current match result."""

    states: dict[str, TeamState] = {}
    h2h_states: dict[tuple[str, str], HeadToHeadState] = {}
    referee_states: dict[str, RefereeState] = {}
    feature_rows: list[dict[str, float]] = []
    labels: list[str | None] = []
    meta_rows: list[dict[str, Any]] = []

    ordered = matches.sort_values(["utc_date", "match_id"]).reset_index(drop=True)

    for _, row in ordered.iterrows():
        home_key = _team_key(row, "home")
        away_key = _team_key(row, "away")
        home = states.setdefault(home_key, TeamState())
        away = states.setdefault(away_key, TeamState())
        pair_key = _pair_key(home_key, away_key)
        h2h = h2h_states.setdefault(pair_key, HeadToHeadState())
        referee_key = _referee_key(row)
        referee = referee_states.get(referee_key) if referee_key else None
        match_date = row.get("utc_date")

        should_emit = include_unfinished or _finished_with_score(row)
        if should_emit:
            home_features = _features_for_team(home, "home", match_date, last_n)
            away_features = _features_for_team(away, "away", match_date, last_n)
            elo_diff = (home.elo + 65.0) - away.elo
            points_diff = (
                home_features["home_all_points_per_game"]
                - away_features["away_all_points_per_game"]
            )
            goal_diff = (
                home_features["home_all_goal_diff"] - away_features["away_all_goal_diff"]
            )
            last_points_diff = (
                home_features[f"home_last{last_n}_points"]
                - away_features[f"away_last{last_n}_points"]
            )
            last_goal_diff = (
                home_features[f"home_last{last_n}_goal_diff"]
                - away_features[f"away_last{last_n}_goal_diff"]
            )
            venue_points_diff = (
                home_features["home_at_home_all_points_per_game"]
                - away_features["away_at_away_all_points_per_game"]
            )
            venue_goal_diff = (
                home_features["home_at_home_all_goal_diff"]
                - away_features["away_at_away_all_goal_diff"]
            )
            home_is_favorite = 1.0 if elo_diff >= 0.0 else 0.0
            favorite_recent_ga = (
                home_features[f"home_last{last_n}_goals_against"]
                if home_is_favorite
                else away_features[f"away_last{last_n}_goals_against"]
            )
            underdog_recent_gf = (
                away_features[f"away_last{last_n}_goals_for"]
                if home_is_favorite
                else home_features[f"home_last{last_n}_goals_for"]
            )
            favorite_venue_ga = (
                home_features[f"home_at_home_last{last_n}_goals_against"]
                if home_is_favorite
                else away_features[f"away_at_away_last{last_n}_goals_against"]
            )
            underdog_venue_gf = (
                away_features[f"away_at_away_last{last_n}_goals_for"]
                if home_is_favorite
                else home_features[f"home_at_home_last{last_n}_goals_for"]
            )
            combined_draw_rate = (
                home_features["home_all_draw_rate"] + away_features["away_all_draw_rate"]
            ) / 2.0
            combined_venue_draw_rate = (
                home_features["home_at_home_all_draw_rate"]
                + away_features["away_at_away_all_draw_rate"]
            ) / 2.0
            combined_recent_draw_rate = (
                home_features[f"home_last{last_n}_draw_rate"]
                + away_features[f"away_last{last_n}_draw_rate"]
            ) / 2.0
            h2h_feature_values = _h2h_features(
                h2h=h2h,
                home_key=home_key,
                away_key=away_key,
                canonical_home_key=pair_key[0],
                last_n=last_n,
            )
            row_features = {
                **_date_features(row),
                **_referee_features(referee),
                **home_features,
                **away_features,
                **h2h_feature_values,
            }
            row_features.update(
                {
                    "elo_diff": elo_diff,
                    "abs_elo_diff": abs(elo_diff),
                    f"last{last_n}_points_diff": last_points_diff,
                    f"abs_last{last_n}_points_diff": abs(last_points_diff),
                    f"last{last_n}_goal_diff_diff": last_goal_diff,
                    f"abs_last{last_n}_goal_diff_diff": abs(last_goal_diff),
                    "all_points_per_game_diff": points_diff,
                    "abs_all_points_per_game_diff": abs(points_diff),
                    "all_goals_for_diff": home_features["home_all_goals_for"]
                    - away_features["away_all_goals_for"],
                    "all_goals_against_diff": home_features[
                        "home_all_goals_against"
                    ]
                    - away_features["away_all_goals_against"],
                    "all_goal_diff_diff": goal_diff,
                    "abs_all_goal_diff_diff": abs(goal_diff),
                    f"venue_last{last_n}_points_diff": home_features[
                        f"home_at_home_last{last_n}_points"
                    ]
                    - away_features[f"away_at_away_last{last_n}_points"],
                    "venue_all_points_per_game_diff": venue_points_diff,
                    "abs_venue_all_points_per_game_diff": abs(venue_points_diff),
                    "venue_all_goal_diff_diff": venue_goal_diff,
                    "abs_venue_all_goal_diff_diff": abs(venue_goal_diff),
                    "combined_draw_rate": combined_draw_rate,
                    "combined_venue_draw_rate": combined_venue_draw_rate,
                    f"combined_last{last_n}_draw_rate": combined_recent_draw_rate,
                    "favorite_recent_goals_against": favorite_recent_ga,
                    "underdog_recent_goals_for": underdog_recent_gf,
                    "favorite_venue_goals_against": favorite_venue_ga,
                    "underdog_venue_goals_for": underdog_venue_gf,
                    "favorite_concede_underdog_score_risk": (
                        favorite_recent_ga + underdog_recent_gf
                    )
                    / 2.0,
                    "favorite_venue_concede_underdog_score_risk": (
                        favorite_venue_ga + underdog_venue_gf
                    )
                    / 2.0,
                    "rest_days_diff": home_features["home_rest_days"]
                    - away_features["away_rest_days"],
                    "played_diff": home.played - away.played,
                    "home_advantage": 1.0,
                }
            )
            feature_rows.append(row_features)
            labels.append(row.get("outcome") if _finished_with_score(row) else None)
            meta_rows.append(
                {
                    "match_id": row.get("match_id"),
                    "utc_date": match_date,
                    "status": row.get("status"),
                    "home_team": row.get("home_team"),
                    "away_team": row.get("away_team"),
                    "season_start_year": row.get("season_start_year"),
                    "matchday": row.get("matchday"),
                    "referee_name": row.get("referee_name"),
                    "outcome": row.get("outcome") if _finished_with_score(row) else None,
                }
            )

        if _finished_with_score(row):
            home_goals = int(row["home_goals"])
            away_goals = int(row["away_goals"])
            _update_elo(home, away, home_goals, away_goals)
            _update_team(home, match_date, home_goals, away_goals, last_n, "home")
            _update_team(away, match_date, away_goals, home_goals, last_n, "away")
            _update_h2h(
                state=h2h,
                home_key=home_key,
                away_key=away_key,
                canonical_home_key=pair_key[0],
                home_goals=home_goals,
                away_goals=away_goals,
                last_n=last_n,
            )
            if referee_key:
                referee_state = referee_states.setdefault(referee_key, RefereeState())
                _update_referee(
                    referee_state, home_goals, away_goals, str(row.get("outcome"))
                )

    return pd.DataFrame(feature_rows), pd.Series(labels, name="outcome"), pd.DataFrame(meta_rows)
