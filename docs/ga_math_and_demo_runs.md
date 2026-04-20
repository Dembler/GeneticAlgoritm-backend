# Математика Маршрутизации И Параметры Запуска

## 1. Что оптимизирует система

Система решает задачу перестановки точек маршрута при фиксированных или свободных концах.

Если:

- задано `n` точек,
- `fix_ends = true`,

то генетический алгоритм переставляет только внутренние точки:

```text
[0] + permutation(1..n-2) + [n-1]
```

Если:

- `fix_ends = false`,

то геномом является перестановка всех точек:

```text
permutation(0..n-1)
```

Источники:

- [RouteRequest](/c:/Users/ndobrosockih/PycharmProjects/GeneticAlgoritm/app/domain/models.py#L113)
- [decode genome](/c:/Users/ndobrosockih/PycharmProjects/GeneticAlgoritm/app/services/route_optimizer.py#L343)

## 2. Как строятся данные для оптимизатора

Перед запуском GA собирается контекст задачи:

- матрица расстояний `distance_matrix_km`
- матрица времени `duration_matrix_min`
- матрица трафика `traffic_matrix`
- матрица платных дорог `toll_matrix`
- погодный снимок
- высотный профиль

Это делается в [ContextService](/c:/Users/ndobrosockih/PycharmProjects/GeneticAlgoritm/app/services/context_service.py#L52).

В терминах математики маршрут задаётся как последовательность индексов:

```text
order = [i_0, i_1, ..., i_k]
```

Для каждого ребра `i_t -> i_(t+1)` берутся значения из матриц:

```text
d_t = distance_matrix[i_t][i_(t+1)]
tau_t = duration_matrix[i_t][i_(t+1)]
c_t = traffic_matrix[i_t][i_(t+1)]
tol_t = toll_matrix[i_t][i_(t+1)]
```

## 3. Математика критериев

Основной расчёт кандидата выполняется в [CriteriaService.evaluate(...)](/c:/Users/ndobrosockih/PycharmProjects/GeneticAlgoritm/app/services/criteria_service.py#L25).

### 3.1. Время прохождения сегмента

Пусть:

- `tau_t` — базовое время из матрицы маршрутизации
- `c_t` — индекс загруженности на сегменте
- `w` — severity погоды

Тогда модель времени:

```text
weather_delay_factor = 0.35 * w
duration_t = tau_t * (1 + c_t + weather_delay_factor)
```

Если матрица времени не дала положительного значения, используется fallback:

```text
duration_t = distance_t / 38 * 60
```

### 3.2. Риски надёжности и безопасности

Для каждого сегмента:

```text
incident_proxy = clamp(0.1 + 0.25*c_t + 0.2*w, 0, 1)

reliability_risk_t =
  clamp(0.45*c_t + 0.35*w + 0.2*incident_proxy, 0, 1)

safety_risk_t =
  clamp(0.5*w + 0.3*c_t + 0.2*night_factor, 0, 1)
```

Где:

- `night_factor = 1.0` ночью
- `night_factor = 0.5` в переходные часы
- `night_factor = 0.1` днём

Итог по маршруту:

```text
congestion_index = average(c_t)
reliability_risk_avg = average(reliability_risk_t)
safety_risk_avg = average(safety_risk_t)
reliability_score = 1 - reliability_risk_avg
```

### 3.3. Рельеф и уклон

Рельеф больше не оценивается только по двум концам отрезка. Для каждого ребра `i -> j`
строится набор промежуточных elevation-сэмплов между точками, после чего набор и сброс
высоты считаются накопительно по всему профилю.

Для сэмплов `p_0, p_1, ..., p_m`:

```text
delta_h_r = elevation(p_(r+1)) - elevation(p_r)
gain_r = max(delta_h_r, 0)
loss_r = max(-delta_h_r, 0)
```

Для одного сегмента маршрута:

```text
gain_t = sum(gain_r)
loss_t = sum(loss_r)
mean_elevation_t = weighted_average(elevation, by horizontal distance)
```

По всему маршруту:

```text
total_gain = sum(gain_t)
total_loss = sum(loss_t)
horizontal_m = total_distance_km * 1000

uphill_pct = min(100, total_gain / horizontal_m * 100)
downhill_pct = min(100, total_loss / horizontal_m * 100)
```

Таким образом, если у двух опорных точек одинаковая высота, но между ними есть перевалы
и спуски, они всё равно попадут в `total_gain` и `total_loss`.

На карте рельеф показывается отдельно по дорожной геометрии уже построенного маршрута:

- оранжевый участок — подъём
- бирюзовый участок — спуск
- серый участок — почти ровный фрагмент

### 3.4. Топливная модель

Расчёты находятся в [FuelCostService](/c:/Users/ndobrosockih/PycharmProjects/GeneticAlgoritm/app/services/fuel_cost.py#L154).

Пусть:

- `q_base` — базовый расход, л/100 км
- `D` — расстояние маршрута
- `M_terrain` — поправка на уклон
- `M_mountain` — поправка на высоту и горный режим
- `M_temp` — поправка на температуру
- `M_congestion = 1 + 0.2 * congestion_index`

Тогда:

```text
liters =
  D * q_base / 100
  * M_terrain
  * M_mountain
  * M_temp
  * M_congestion
```

Стоимость:

```text
fuel_cost = liters * price_per_liter + total_toll
```

Выбросы:

```text
co2_kg = liters * emission_factor
```

где:

- бензин: `2.31 кг CO2 / л`
- дизель: `2.68 кг CO2 / л`

### 3.5. Ограничения и штрафы

Если маршрут нарушает ограничения, начисляется penalty:

```text
distance penalty = 1200 * relative_violation
duration penalty = 1200 * relative_violation
fuel penalty     = 1400 * relative_violation
co2 penalty      = 1100 * relative_violation
safety penalty   = 1600 * relative_violation
```

Например для топлива:

```text
if fuel_cost > max_fuel_cost:
  penalty += 1400 * (fuel_cost - max_fuel_cost) / max_fuel_cost
```

Маршрут считается допустимым, если:

```text
feasible = penalty <= 1e-9
```

## 4. Как считается objective score

### 4.1. Для одного кандидата

Если в сравнении только один маршрут, используется абсолютная взвешенная сумма:

```text
score =
  w_distance * distance
  + w_duration * duration
  + w_fuel * fuel_cost
  + w_emissions * co2
  + w_congestion * congestion
  + w_weather * weather_risk
  + w_reliability * (1 - reliability_score)
  + w_safety * safety_risk
  + w_tolls * toll_cost
  + penalty
```

Источник: [absolute weighted score](/c:/Users/ndobrosockih/PycharmProjects/GeneticAlgoritm/app/services/criteria_service.py#L221)

### 4.2. Для популяции

Если кандидатов несколько, сначала идёт min-max нормировка внутри текущего набора:

```text
norm(x) = (x - min_population) / (max_population - min_population)
```

После этого:

```text
score =
  w_distance * norm(distance)
  + ...
  + w_tolls * norm(toll_cost)
  + norm(penalty)
```

Источник: [population normalization](/c:/Users/ndobrosockih/PycharmProjects/GeneticAlgoritm/app/services/criteria_service.py#L173)

Важно:

- это относительный score
- его корректно сравнивать внутри одного запуска
- его нельзя напрямую сравнивать между разными экспериментами

## 5. Как работает генетический алгоритм

Источник: [RouteOptimizer.optimize(...)](/c:/Users/ndobrosockih/PycharmProjects/GeneticAlgoritm/app/services/route_optimizer.py#L58)

### 5.1. Инициализация

Начальная популяция состоит из:

- исходного порядка
- `nearest-neighbor` seed
- случайных перестановок

Источник: [initial population](/c:/Users/ndobrosockih/PycharmProjects/GeneticAlgoritm/app/services/route_optimizer.py#L284)

### 5.2. Кэш вычислений

Если одна и та же перестановка встречается снова, оценка берётся из cache:

```text
eval_cache[genome] -> CandidateEvaluation
```

Это снижает число реальных вычислений.

### 5.3. Selection

В `weighted`-режиме:

- турнир между двумя кандидатами
- побеждает feasible
- затем меньший `objective_score`

Источник: [weighted tournament](/c:/Users/ndobrosockih/PycharmProjects/GeneticAlgoritm/app/services/route_optimizer.py#L438)

В `pareto`-режиме:

- сначала меньший `rank`
- затем больший `crowding`
- затем меньший `objective_score`

Источник: [pareto tournament](/c:/Users/ndobrosockih/PycharmProjects/GeneticAlgoritm/app/services/route_optimizer.py#L425)

### 5.4. Crossover

Используется `order crossover`:

- случайно выбирается сегмент
- сегмент копируется от первого родителя
- остальные гены дозаполняются в порядке второго родителя

Источник: [order crossover](/c:/Users/ndobrosockih/PycharmProjects/GeneticAlgoritm/app/services/route_optimizer.py#L449)

### 5.5. Mutation

С вероятностью `mutation_rate` применяется:

- swap двух генов
- или reverse подотрезка

Источник: [mutation](/c:/Users/ndobrosockih/PycharmProjects/GeneticAlgoritm/app/services/route_optimizer.py#L474)

## 6. Как работает Pareto / NSGA-II часть

Вектор доминирования:

```text
(
  distance + penalty,
  duration + penalty,
  fuel_cost + penalty,
  co2 + penalty,
  congestion + penalty,
  weather_risk + penalty,
  reliability_risk + penalty,
  safety_risk + penalty,
  toll_cost + penalty
)
```

Источник: [dominance vector](/c:/Users/ndobrosockih/PycharmProjects/GeneticAlgoritm/app/services/criteria_service.py#L159)

Дальше:

- строятся фронты доминирования
- вычисляется crowding distance
- новое поколение набирается по фронтам и crowding

Источники:

- [non-dominated sorting](/c:/Users/ndobrosockih/PycharmProjects/GeneticAlgoritm/app/services/route_optimizer.py#L350)
- [crowding](/c:/Users/ndobrosockih/PycharmProjects/GeneticAlgoritm/app/services/route_optimizer.py#L388)

## 7. Динамические веса

Источник: [DynamicWeightsService](/c:/Users/ndobrosockih/PycharmProjects/GeneticAlgoritm/app/services/dynamic_weights_service.py#L8)

Сначала берутся нормированные базовые веса, затем применяются мультипликаторы.

### Профили приоритета

```text
fastest:
  duration *= 1.38
  congestion *= 1.20

cheapest:
  fuel_cost *= 1.42
  tolls *= 1.22

safest:
  safety *= 1.45
  reliability *= 1.30
  weather_risk *= 1.10

greenest:
  emissions *= 1.55
  fuel_cost *= 1.10
```

### Контекстные триггеры

```text
peak_hour:
  duration *= 1.28
  congestion *= 1.22
  reliability *= 1.12

bad_weather:
  weather_risk *= 1.35
  safety *= 1.22
  reliability *= 1.15

high_congestion:
  duration *= 1.16
  congestion *= 1.30
  reliability *= 1.13

high_fuel_price:
  fuel_cost *= 1.26
  emissions *= 1.10
```

После этого веса снова нормируются.

## 8. Post-analysis: segment insights и stress test

Источник: [RouteAnalysisService](/c:/Users/ndobrosockih/PycharmProjects/GeneticAlgoritm/app/services/route_analysis_service.py#L17)

### Segment insights

Для каждого сегмента считается доминирующий фактор:

- weather
- congestion
- elevation
- tolls
- safety
- reliability

Результат нужен для объяснимости маршрута.

### Stress test

Делается `120` Монте-Карло симуляций.

На каждом сегменте случайно варьируются эффекты:

```text
weather
congestion
elevation
tolls
safety
reliability
```

По симуляциям считаются:

- вероятность уложиться во время
- вероятность уложиться в бюджет
- вероятность уложиться по safety
- failure probability
- resilience index

## 9. Чёткие параметры для запуска системы

## Backend

Из каталога backend:

```powershell
cd C:\Users\ndobrosockih\PycharmProjects\GeneticAlgoritm
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Backend будет доступен по:

- `http://127.0.0.1:8000`
- OpenAPI: `http://127.0.0.1:8000/docs`

## Frontend

Из каталога frontend:

```powershell
cd C:\Users\ndobrosockih\PycharmProjects\GeneticAlgoritm-frontend
npm install
$env:VITE_API_TARGET="http://127.0.0.1:8000"
npm run dev
```

Frontend будет доступен по:

- `http://127.0.0.1:5173`

## Production build фронтенда

```powershell
cd C:\Users\ndobrosockih\PycharmProjects\GeneticAlgoritm-frontend
npm run build
```

Сборка уходит в:

- `C:\Users\ndobrosockih\PycharmProjects\GeneticAlgoritm\app\frontend_dist`

## 10. Параметры для демонстрационных запусков

Ниже даны готовые параметры, которые можно вставлять в UI или отправлять на `POST /api/routes`.

## Сценарий A. Базовая перестановка

Цель:

- показать, что алгоритм реально переставляет внутренние точки
- показать выигрыш по distance и duration

```json
{
  "points": [
    {"lat": 51.67, "lon": 39.18, "label": "Start"},
    {"lat": 51.70, "lon": 39.30, "label": "FarNorth"},
    {"lat": 51.66, "lon": 39.22, "label": "EastCluster"},
    {"lat": 51.69, "lon": 39.24, "label": "CentralDepot"},
    {"lat": 51.65, "lon": 39.16, "label": "Finish"}
  ],
  "optimize": true,
  "fix_ends": true,
  "profile": "driving",
  "vehicle_class": "passenger",
  "fuel_type": "petrol",
  "fuel_consumption_l_per_100km": 8.5,
  "optimize_mode": "weighted",
  "priority_profile": "balanced",
  "use_dynamic_weights": true,
  "departure_at": "2026-04-20T12:00:00Z",
  "population_size": 24,
  "generations": 40,
  "crossover_rate": 0.88,
  "mutation_rate": 0.22,
  "max_alternatives": 8,
  "random_seed": 7
}
```

Ожидаемый эффект:

- алгоритм должен переставить внутренние точки
- distance и duration должны заметно уменьшиться

## Сценарий B. Trade-off по стоимости

Цель:

- показать конфликт `быстро` против `дёшево`
- показать разницу между raw-метриками и relative objective score

```json
{
  "points": [
    {"lat": 0.0, "lon": 0.0, "label": "Start"},
    {"lat": 0.0, "lon": 1.0, "label": "TolledShort"},
    {"lat": 1.0, "lon": 0.0, "label": "FreeMid"},
    {"lat": 1.0, "lon": 1.0, "label": "Connector"},
    {"lat": 2.0, "lon": 1.0, "label": "Finish"}
  ],
  "optimize": true,
  "fix_ends": true,
  "profile": "driving",
  "vehicle_class": "passenger",
  "fuel_type": "petrol",
  "fuel_consumption_l_per_100km": 8.5,
  "optimize_mode": "weighted",
  "priority_profile": "cheapest",
  "use_dynamic_weights": true,
  "departure_at": "2026-04-20T09:00:00Z",
  "population_size": 24,
  "generations": 40,
  "crossover_rate": 0.88,
  "mutation_rate": 0.22,
  "max_alternatives": 8,
  "random_seed": 4
}
```

Что смотреть:

- cost
- duration
- Pareto-альтернативы
- baseline vs optimized

## Сценарий C. Стрессовый и infeasible кейс

Цель:

- показать штрафы
- показать `feasible = false`
- показать stress test и dominant factors

```json
{
  "points": [
    {"lat": 0.0, "lon": 0.0, "label": "Start"},
    {"lat": 0.0, "lon": 1.0, "label": "UrbanCluster"},
    {"lat": 1.0, "lon": 0.0, "label": "HillSection"},
    {"lat": 1.0, "lon": 1.0, "label": "IndustrialZone"},
    {"lat": 2.0, "lon": 1.0, "label": "Finish"}
  ],
  "optimize": true,
  "fix_ends": true,
  "profile": "driving",
  "vehicle_class": "passenger",
  "fuel_type": "petrol",
  "fuel_consumption_l_per_100km": 8.5,
  "optimize_mode": "weighted",
  "priority_profile": "safest",
  "use_dynamic_weights": true,
  "departure_at": "2026-01-15T08:00:00Z",
  "constraints": {
    "max_duration_min": 70,
    "max_fuel_cost": 140,
    "max_safety_risk": 0.32
  },
  "population_size": 24,
  "generations": 40,
  "crossover_rate": 0.88,
  "mutation_rate": 0.22,
  "max_alternatives": 8,
  "random_seed": 12
}
```

Что смотреть:

- `constraint_penalty`
- `feasible`
- `stress_test`
- `segment_insights`

## Сценарий D. Живой запуск с внешними репозиториями

Цель:

- показать работу на реальных API и реальных данных маршрутизации

```json
{
  "points": [
    {"lat": 51.6720, "lon": 39.1843, "label": "Start"},
    {"lat": 51.7065, "lon": 39.2089, "label": "North"},
    {"lat": 51.6580, "lon": 39.2480, "label": "East"},
    {"lat": 51.6405, "lon": 39.1625, "label": "Finish"}
  ],
  "optimize": true,
  "fix_ends": true,
  "profile": "driving",
  "vehicle_class": "passenger",
  "fuel_type": "petrol",
  "fuel_consumption_l_per_100km": 8.5,
  "optimize_mode": "pareto",
  "priority_profile": "balanced",
  "use_dynamic_weights": true,
  "departure_at": "2026-04-20T12:00:00Z",
  "population_size": 24,
  "generations": 40,
  "crossover_rate": 0.88,
  "mutation_rate": 0.22,
  "max_alternatives": 8,
  "random_seed": 21
}
```

Что показывать:

- `provider`
- `data_sources`
- `alternatives`
- `comparison`

## 11. Что лучше проговаривать на защите

- Сначала показать формирование контекста задачи.
- Затем объяснить, что геном — это перестановка индексов точек.
- Потом показать формулы времени, стоимости топлива, штрафов и итогового score.
- После этого показать `weighted` и `pareto` режимы.
- Завершить демонстрацией stress test и объяснимости через `segment_insights`.
