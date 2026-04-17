# Route Optimization Lab

Multi-criteria route optimization platform with FastAPI backend and separate Vue 3 frontend.

## Project structure
- `app/` - backend on FastAPI, optimization services, API, persistence, and production serving of built frontend assets.
- `frontend/` - separate frontend project on Vue 3 + TypeScript + Vite.
- `app/frontend_dist/` - production build output from Vite, served by FastAPI at `/`.

## Features
- Route optimization with weighted scoring + Pareto front.
- Metrics: distance, duration, fuel, CO2, congestion, weather risk, reliability, safety, tolls.
- Automatic weather and elevation ingestion from Open-Meteo.
- Automatic slope-aware fuel correction and temperature correction.
- Optional toll-cost matrix from TollGuru API with fallback.
- Run history in SQLite and CSV export.

## Run
1. Install backend dependencies:
```bash
pip install -r requirements.txt
```
2. Install frontend dependencies:
```bash
cd frontend
npm install
cd ..
```
3. Start backend:
```bash
uvicorn app.main:app --reload
```
4. For frontend development, run Vite in a second terminal:
```bash
cd frontend
npm run dev
```
Vite proxies `/api` to `http://127.0.0.1:8000`.

5. Open:
- Frontend dev UI: `http://127.0.0.1:5173`
- UI: `http://127.0.0.1:8000`
- OpenAPI: `http://127.0.0.1:8000/docs`

## Production build
Build the Vue frontend into `app/frontend_dist`:

```bash
cd frontend
npm run build
```

Then start FastAPI:

```bash
uvicorn app.main:app
```

FastAPI serves the built SPA from `app/frontend_dist`. If the frontend is not built yet, the root path returns a `503` with an explanatory message.

## API
- `POST /api/routes` - build and optimize route.
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

### Weather and elevation (Open-Meteo)
- `APP_WEATHER_ENABLED=true|false`
- `APP_ELEVATION_ENABLED=true|false`
- `APP_OPENMETEO_BASE_URL=https://api.open-meteo.com`

### Toll roads (TollGuru)
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
- Traffic API integrations are disabled now; source is `traffic-disabled`.
- Weather source: Open-Meteo Forecast API.
- Elevation source: Open-Meteo Elevation API.
- Slope shares (uphill/downhill) are calculated automatically from elevation data.
- Toll source: TollGuru when enabled and key is valid; otherwise fallback source is `toll-disabled`.

PowerShell example:
```powershell
$env:APP_WEATHER_ENABLED="true"
$env:APP_ELEVATION_ENABLED="true"
$env:APP_OPENMETEO_BASE_URL="https://api.open-meteo.com"
$env:APP_TOLL_ENABLED="true"
$env:APP_TOLL_API_KEY="<YOUR_TOLLGURU_API_KEY>"
$env:APP_TOLL_BASE_URL="https://apis.tollguru.com/toll"
```

After changing env vars, restart `uvicorn`.

Check response sources in:
- `data_sources.weather`
- `data_sources.elevation`
- `data_sources.traffic`
- `data_sources.tolls`
- `data_sources.fuel_prices`

## Temperature impact on fuel
Fuel consumption includes temperature multiplier in `FuelCostService.temperature_multiplier(...)`.

Reference sources:
- https://www.energy.gov/energysaver/fuel-economy-cold-weather
- https://www.fueleconomy.gov/feg/hotweather.shtml
- https://www.fueleconomy.gov/feg/coldweather.shtml

## Optimizer behavior notes
- Real point-order optimization starts only when there are enough points to permute:
  - `fix_ends=true` requires at least 4 points.
  - `fix_ends=false` requires at least 3 points.
- `use_dynamic_weights=false` disables context triggers (weather/traffic/peak/fuel price), but still keeps the selected `priority_profile`.
- Diagnostics now expose:
  - `optimization_active`
  - `optimization_reason` (`not_enough_points`, `fixed_route`, `optimize_disabled`)
  - `score_mode` (`absolute_single_candidate`, `population_normalized`)
- UI now shows a visual comparison block:
  - route and metrics `Before GA` vs `After GA`
  - deltas and improvement percentages
  - score decomposition by criterion (`weight`, `raw`, `norm`, `contribution`)
  - compact human-readable summary of what improved
  - duplicate Pareto rows are collapsed for readability

## Model assumptions
The implementation uses a hybrid engineering approach:
- Mountain and altitude fuel multipliers are calibrated with published mountain-condition studies.
- Temperature correction follows DOE/FuelEconomy public guidance.
- Priority profiles and dynamic weights are rule-based and tuned for robust behavior in web-routing scenarios.
- See `docs/model_assumptions.md` for details and scope boundaries.
