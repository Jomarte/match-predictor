# Match Predictor

Python football match predictor using `football-data.org` v4 data. The default competition is Portugal's Primeira Liga, code `PPL`.

It fetches and caches matches, builds rolling "last games" features, trains a multiclass model for home win / draw / away win, and predicts upcoming or manually supplied fixtures.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
Copy-Item .env.example .env
```

Put your API key in `.env`:

```text
FOOTBALL_DATA_API_KEY=your_real_key
```

## Typical Workflow

Fetch historical Primeira Liga seasons:

```powershell
match-predictor fetch --seasons 2023 2024 2025 --status FINISHED
```

Train the model:

```powershell
match-predictor train
```

Fetch current scheduled fixtures and predict them:

```powershell
match-predictor predict-upcoming --season 2026
```

Predict one manual fixture from cached history:

```powershell
match-predictor predict-match --home "SL Benfica" --away "FC Porto"
```

## Useful Competition Codes

- `PPL` - Primeira Liga
- `PL` - Premier League
- `PD` - La Liga
- `BL1` - Bundesliga
- `SA` - Serie A
- `FL1` - Ligue 1
- `DED` - Eredivisie
- `CL` - Champions League

Availability depends on your football-data.org plan.

## Notes

- Data is cached under `data/raw` so repeated training does not hammer the API.
- The free football-data.org plan is rate limited, so the fetch command sleeps between API requests.
- Model artifacts are saved under `models`.
- Predictions are also exported under `data/predictions`.
