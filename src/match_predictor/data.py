from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PROCESSED_COLUMNS = [
    "match_id",
    "competition",
    "season_start_year",
    "utc_date",
    "status",
    "matchday",
    "stage",
    "score_winner",
    "score_duration",
    "home_team_id",
    "home_team",
    "home_team_short_name",
    "home_team_tla",
    "away_team_id",
    "away_team",
    "away_team_short_name",
    "away_team_tla",
    "home_goals",
    "away_goals",
    "home_half_time_goals",
    "away_half_time_goals",
    "outcome",
    "referee_id",
    "referee_name",
    "referee_nationality",
    "has_referee",
    "last_updated",
]


def _full_time_score(match: dict[str, Any]) -> tuple[int | None, int | None]:
    score = match.get("score") or {}
    full_time = score.get("fullTime") or {}
    return full_time.get("home"), full_time.get("away")


def _half_time_score(match: dict[str, Any]) -> tuple[int | None, int | None]:
    score = match.get("score") or {}
    half_time = score.get("halfTime") or {}
    return half_time.get("home"), half_time.get("away")


def _main_referee(match: dict[str, Any]) -> dict[str, Any]:
    referees = match.get("referees") or []
    for referee in referees:
        if referee.get("type") == "REFEREE":
            return referee
    return referees[0] if referees else {}


def _outcome(home_goals: int | None, away_goals: int | None) -> str | None:
    if home_goals is None or away_goals is None:
        return None
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def _season_start_year(match: dict[str, Any], fallback: Any = None) -> int | None:
    season = match.get("season") or {}
    start_date = season.get("startDate")
    if start_date:
        return int(start_date[:4])
    if fallback is not None:
        try:
            return int(fallback)
        except (TypeError, ValueError):
            return None
    return None


def normalise_matches(payloads: Iterable[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for payload in payloads:
        competition = (payload.get("competition") or {}).get("code")
        fallback_season = (payload.get("filters") or {}).get("season")

        for match in payload.get("matches", []):
            home_goals, away_goals = _full_time_score(match)
            home_half_time_goals, away_half_time_goals = _half_time_score(match)
            home_team = match.get("homeTeam") or {}
            away_team = match.get("awayTeam") or {}
            score = match.get("score") or {}
            referee = _main_referee(match)
            match_competition = (match.get("competition") or {}).get("code") or competition

            rows.append(
                {
                    "match_id": match.get("id"),
                    "competition": match_competition,
                    "season_start_year": _season_start_year(match, fallback_season),
                    "utc_date": match.get("utcDate"),
                    "status": match.get("status"),
                    "matchday": match.get("matchday"),
                    "stage": match.get("stage"),
                    "score_winner": score.get("winner"),
                    "score_duration": score.get("duration"),
                    "home_team_id": home_team.get("id"),
                    "home_team": home_team.get("name"),
                    "home_team_short_name": home_team.get("shortName"),
                    "home_team_tla": home_team.get("tla"),
                    "away_team_id": away_team.get("id"),
                    "away_team": away_team.get("name"),
                    "away_team_short_name": away_team.get("shortName"),
                    "away_team_tla": away_team.get("tla"),
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "home_half_time_goals": home_half_time_goals,
                    "away_half_time_goals": away_half_time_goals,
                    "outcome": _outcome(home_goals, away_goals),
                    "referee_id": referee.get("id"),
                    "referee_name": referee.get("name"),
                    "referee_nationality": referee.get("nationality"),
                    "has_referee": bool(referee),
                    "last_updated": match.get("lastUpdated"),
                }
            )

    frame = pd.DataFrame(rows, columns=PROCESSED_COLUMNS)
    if frame.empty:
        return frame

    frame = frame.drop_duplicates("match_id", keep="last")
    frame["utc_date"] = pd.to_datetime(frame["utc_date"], utc=True, errors="coerce")
    frame["last_updated"] = pd.to_datetime(
        frame["last_updated"], utc=True, errors="coerce"
    )
    frame = frame.sort_values(["utc_date", "match_id"]).reset_index(drop=True)
    return frame


def load_raw_payloads(raw_dir: Path, competition: str) -> list[dict[str, Any]]:
    payloads = []
    for path in sorted(raw_dir.glob(f"{competition.upper()}_*.json")):
        payloads.append(json.loads(path.read_text(encoding="utf-8-sig")))
    return payloads


def rebuild_processed_csv(data_dir: Path, competition: str) -> Path:
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    frame = normalise_matches(load_raw_payloads(raw_dir, competition))
    output_path = processed_dir / f"{competition.upper()}_matches.csv"
    frame.to_csv(output_path, index=False)
    return output_path


def load_processed_matches(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    frame["utc_date"] = pd.to_datetime(frame["utc_date"], utc=True, errors="coerce")
    if "last_updated" in frame.columns:
        frame["last_updated"] = pd.to_datetime(
            frame["last_updated"], utc=True, errors="coerce"
        )
    return frame.sort_values(["utc_date", "match_id"]).reset_index(drop=True)
