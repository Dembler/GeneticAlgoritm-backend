# Route Optimization Lab Backend

FastAPI backend for a multi-criteria route optimization platform. The Vue 3 frontend lives in a separate repository and is not stored in this backend repo.

## Repository boundaries
- This repository contains only backend code, backend tests, and backend documentation.
- Frontend source code must stay in a separate repository.
- `app/frontend_dist/` is only a generated production bundle copied from the frontend build and served by FastAPI at `/`.
- `app/frontend_dist/` is ignored by git and must not be edited as source code here.

## Project structure
- `app/` - FastAPI app, API routes, services, repositories, domain models, and SPA serving entrypoint.
- `docs/` - backend-side notes and model assumptions.
- `tests/` - backend tests.
- `data/` - local runtime data such as SQLite database files.

## Run backend
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Start backend:
```bash
uvicorn app.main:app --reload
```
3. Open:
- API docs: `http://127.0.0.1:8000/docs`
- UI through built frontend bundle: `http://127.0.0.1:8000`

## Frontend integration
- During frontend development, run the separate frontend repository in its own terminal.
- The frontend dev server should proxy `/api` to `http://127.0.0.1:8000`.
- For production-style local integration, build the frontend so that output is written into `app/frontend_dist/`.

If `app/frontend_dist/` is missing, the backend returns `503` on `/` with an explanatory message.

## API
- `POST /api/routes` - build and optimize a route.
- `GET /api/runs?limit=20` - recent runs.
- `GET /api/runs/{run_id}` - run details.
- `GET /api/runs/{run_id}/report.csv` - CSV report.

## Configuration (`APP_*`)
### Routing
- `APP_OSRM_ENABLED=true|false`
- `APP_OSRM_BASE_URL=https://router.project-osrm.org`
- `APP_REQUEST_TIMEOUT_SEC=10.0`
- `APP_CACHE_TTL_SEC=300`

### Fuel prices
- `APP_FUEL_PRICE_SOURCE_URL=...`
- `APP_FUEL_PRICE_SOURCE_NAME=Росстат`
- `APP_FUEL_PRICE_CURRENCY=RUB`
- `APP_FUEL_PRICE_CACHE_TTL_SEC=3600`
- `APP_FUEL_PRICE_FALLBACK_PETROL=63.0`
- `APP_FUEL_PRICE_FALLBACK_DIESEL=68.0`

### Weather and elevation
- `APP_WEATHER_ENABLED=true|false`
- `APP_ELEVATION_ENABLED=true|false`
- `APP_OPENMETEO_BASE_URL=https://api.open-meteo.com`

### Toll roads
- `APP_TOLL_ENABLED=true|false`
- `APP_TOLL_BASE_URL=https://apis.tollguru.com/toll`
- `APP_TOLL_API_KEY=...`
- `APP_TOLL_MAX_CONCURRENCY=8`

### Optimizer and persistence
- `APP_GA_DEFAULT_POPULATION=96`
- `APP_GA_DEFAULT_GENERATIONS=120`
- `APP_OPTIMIZER_ENABLE_PARETO=true|false`
- `APP_ROUTE_RUNS_DB_PATH=data/route_runs.db`

## Data source behavior
- Traffic integrations are currently disabled; source is `traffic-disabled`.
- Weather source: Open-Meteo Forecast API.
- Elevation source: Open-Meteo Elevation API.
- Slope shares are calculated automatically from elevation data.
- Toll source: TollGuru when enabled and configured; otherwise `toll-disabled`.

## Model assumptions
The implementation uses a hybrid engineering approach:
- Mountain and altitude fuel multipliers are calibrated with published mountain-condition studies.
- Temperature correction follows DOE/FuelEconomy public guidance.
- Priority profiles and dynamic weights are rule-based and tuned for robust behavior in web-routing scenarios.
- See `docs/model_assumptions.md` for details and scope boundaries.
