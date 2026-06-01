# EdgeShift

Sports betting EV calculator for MLB and NHL. Compares model-predicted win probabilities against market-implied odds to surface positive expected value bets with Kelly-sized stakes.

## Prerequisites

- Python 3.11+
- Node 20+
- PostgreSQL (for MLB engine)

## Backend

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env   # fill in values
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Required env vars** (in `backend/.env`):

| Variable | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string for MLB (`postgresql://user:pass@host/mlb_model`) |
| `ODDS_API_KEY` | [The Odds API](https://the-odds-api.com) key — used by both MLB and NHL engines |
| `NHL_DB_PATH` | Absolute path to `nhl_predictor.db` on the server |
| `ALLOWED_ORIGINS` | Comma-separated frontend origins for CORS |

## Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local   # set API_URL
npm run dev
```

**Required env vars** (in `frontend/.env.local`):

```
API_URL=http://localhost:8000
```

## Engines (cron jobs)

Both engines run as scheduled scripts on the server. Run them from their directories so relative paths resolve:

### MLB (`backend/engine/mlb/`)

```bash
cd backend/engine/mlb
python main_cron.py            # morning: Statcast → multi-book odds → predictions
python main_cron.py --pregame  # pre-game (~90 min before first pitch): refresh odds + re-run predictions
python backfill_weather.py     # one-time: populate weather_cache before first retrain
```

Cron schedule:
- Morning predictions: `30 9 * * *` (9:30 AM ET)
- Pre-game odds refresh: `30 17 * * *` (1:30 PM ET)
- Result updates + CLV: `0 19,21,23 * * *` and `30 1 * * *`

### NHL (`backend/engine/nhl/`)

```bash
cd backend/engine/nhl
python main_cron.py            # daily: schedule → stats → features → predict → odds → EV
python main_cron.py --full     # first run: backfill + train model
python main_cron.py --retrain  # incremental update + retrain model
python main_cron.py --pregame  # pre-game (~90 min before first puck): refresh odds + re-run EV
```

Cron schedule:
- Daily predictions: `0 14 * * *` (10:00 AM ET)
- Pre-game odds refresh: `30 21 * * *` (5:30 PM ET)
- Result updates + CLV: `0 4 * * *` and `30 6 * * *`

## API

`GET /api/health` — health check  
`GET /api/mlb/predictions` — today's MLB games with win probabilities and EV  
`GET /api/mlb/ev-bets` — today's MLB +EV bets (Pinnacle baseline, multi-book best odds, CLV)  
`GET /api/mlb/clv-summary` — rolling MLB closing line value stats  
`GET /api/mlb/history` — historical MLB predictions with accuracy stats  
`GET /api/nhl/predictions` — today's NHL games with win probabilities  
`GET /api/nhl/ev-bets` — today's NHL +EV bets (Pinnacle baseline, multi-book best odds, CLV)  
`GET /api/nhl/clv-summary` — rolling NHL closing line value stats  
`GET /api/nhl/history` — historical NHL predictions with accuracy stats  
