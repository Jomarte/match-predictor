from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


BASE_URL = "https://api.football-data.org/v4"


class FootballDataError(RuntimeError):
    """Raised when football-data.org returns an unusable response."""


class FootballDataClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        timeout: int = 30,
        min_request_interval_seconds: float = 6.2,
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.getenv("FOOTBALL_DATA_API_KEY")
        if not self.api_key:
            raise FootballDataError(
                "Missing FOOTBALL_DATA_API_KEY. Add it to .env or pass --api-key."
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.min_request_interval_seconds = min_request_interval_seconds
        self._last_request_at = 0.0

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        elapsed = time.monotonic() - self._last_request_at
        if self._last_request_at and elapsed < self.min_request_interval_seconds:
            time.sleep(self.min_request_interval_seconds - elapsed)

        url = f"{self.base_url}/{path.lstrip('/')}"
        response = requests.get(
            url,
            headers={"X-Auth-Token": self.api_key},
            params={k: v for k, v in (params or {}).items() if v is not None},
            timeout=self.timeout,
        )
        self._last_request_at = time.monotonic()

        if response.status_code == 429:
            raise FootballDataError("Rate limit hit. Wait a minute and try again.")
        if response.status_code >= 400:
            raise FootballDataError(
                f"football-data.org returned HTTP {response.status_code}: {response.text[:500]}"
            )

        return response.json()


def fetch_competition_matches(
    client: FootballDataClient,
    competition: str,
    season: int | None,
    status: str | None,
    cache_dir: Path,
    force: bool = False,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    season_part = str(season) if season is not None else "current"
    status_part = status or "ALL"
    cache_path = cache_dir / f"{competition.upper()}_{season_part}_{status_part}.json"

    if cache_path.exists() and not force:
        return cache_path

    payload = client.get(
        f"competitions/{competition.upper()}/matches",
        params={"season": season, "status": status},
    )
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return cache_path
