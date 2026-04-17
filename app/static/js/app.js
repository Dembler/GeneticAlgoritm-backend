const map = L.map('map', { zoomControl: true }).setView([51.672, 39.1843], 12);

L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
  maxZoom: 19,
  attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
}).addTo(map);

const pointsContainer = document.getElementById('points');
const pointsCounter = document.getElementById('points-counter');
const addButton = document.getElementById('add-point');
const clearButton = document.getElementById('clear-points');
const buildButton = document.getElementById('build-route');
const optimizeCheckbox = document.getElementById('optimize');
const fixEndsCheckbox = document.getElementById('fix-ends');
const profileSelect = document.getElementById('profile');
const vehicleSelect = document.getElementById('vehicle-class');
const fuelSelect = document.getElementById('fuel-type');
const consumptionInput = document.getElementById('fuel-consumption');
const departureInput = document.getElementById('departure-at');
const useDynamicWeightsInput = document.getElementById('use-dynamic-weights');
const optimizeModeInput = document.getElementById('optimize-mode');
const priorityProfileInput = document.getElementById('priority-profile');
const populationInput = document.getElementById('population-size');
const generationsInput = document.getElementById('generations');
const crossoverInput = document.getElementById('crossover-rate');
const mutationInput = document.getElementById('mutation-rate');
const maxAlternativesInput = document.getElementById('max-alternatives');
const optimizationHintElement = document.getElementById('optimization-hint');
const resultContainer = document.getElementById('result');
const mapOverview = document.getElementById('map-overview');
const historyTable = document.getElementById('history-table');
const historyStats = document.getElementById('history-stats');
const refreshHistoryButton = document.getElementById('refresh-history');
const historyShortcutButton = document.getElementById('history-shortcut');
const paretoModalBody = document.getElementById('pareto-modal-body');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingText = document.getElementById('loading-text');
const optimizationView = document.getElementById('optimization-view');
const historyView = document.getElementById('history-view');
const navButtons = Array.from(document.querySelectorAll('[data-view-target]'));

const weightInputs = {
  distance: document.getElementById('w-distance'),
  duration: document.getElementById('w-duration'),
  fuel_cost: document.getElementById('w-fuel-cost'),
  emissions: document.getElementById('w-emissions'),
  congestion: document.getElementById('w-congestion'),
  weather_risk: document.getElementById('w-weather-risk'),
  reliability: document.getElementById('w-reliability'),
  safety: document.getElementById('w-safety'),
  tolls: document.getElementById('w-tolls'),
};

const constraintInputs = {
  max_distance_km: document.getElementById('c-max-distance'),
  max_duration_min: document.getElementById('c-max-duration'),
  max_fuel_cost: document.getElementById('c-max-fuel-cost'),
  max_co2_kg: document.getElementById('c-max-co2'),
  max_safety_risk: document.getElementById('c-max-safety-risk'),
};

const defaultConsumption = {
  passenger: { petrol: 8.5, diesel: 6.8 },
  light_truck: { petrol: 12.5, diesel: 10.5 },
  heavy_truck: { petrol: 26.0, diesel: 22.0 },
};

let markers = [];
let optimizedLine = null;
let baselineLine = null;
let currentData = null;

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function formatNumber(value, digits = 2) {
  const num = Number(value);
  return Number.isFinite(num) ? num.toFixed(digits) : '—';
}

function formatSigned(value, digits = 2) {
  const num = Number(value);
  if (!Number.isFinite(num)) return '—';
  return `${num > 0 ? '+' : ''}${num.toFixed(digits)}`;
}

function formatDateTime(value) {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '—';
  return date.toLocaleString('ru-RU', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function formatDateShort(value) {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '—';
  return date.toLocaleDateString('ru-RU', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
  });
}

function buildMarkerIcon(type, label) {
  return L.divIcon({
    className: '',
    html: `<div class="route-marker route-marker--${type}"><span>${escapeHtml(label)}</span></div>`,
    iconSize: [32, 32],
    iconAnchor: [16, 28],
    popupAnchor: [0, -28],
  });
}

function showLoading(text) {
  loadingText.textContent = text || 'Выполняется расчёт маршрута.';
  loadingOverlay.classList.remove('is-hidden');
}

function hideLoading() {
  loadingOverlay.classList.add('is-hidden');
}

function openModal(modalId) {
  const modal = document.getElementById(modalId);
  if (!modal) return;
  modal.classList.remove('is-hidden');
  modal.setAttribute('aria-hidden', 'false');
}

function closeModal(modalId) {
  const modal = document.getElementById(modalId);
  if (!modal) return;
  modal.classList.add('is-hidden');
  modal.setAttribute('aria-hidden', 'true');
}

function switchView(view) {
  const isHistory = view === 'history';
  optimizationView.classList.toggle('is-hidden', isHistory);
  historyView.classList.toggle('is-hidden', !isHistory);
  navButtons.forEach((button) => {
    button.classList.toggle('is-active', button.dataset.viewTarget === view);
  });
  if (isHistory) {
    fetchHistory();
  }
}

function updateMapOverview({
  pointsCount = 0,
  status = 'Подготовка',
  headline = 'Добавьте минимум две точки для старта.',
  distance = null,
  duration = null,
} = {}) {
  mapOverview.innerHTML = `
    <div class="map-panel__label">${escapeHtml(status)}</div>
    <div class="map-panel__headline">${escapeHtml(headline)}</div>
    <div class="map-panel__stats">
      <span>Точек: <strong>${pointsCount}</strong></span>
      <span>Дистанция: <strong>${distance === null ? '—' : `${formatNumber(distance, 1)} км`}</strong></span>
      <span>Длительность: <strong>${duration === null ? '—' : `${formatNumber(duration, 0)} мин`}</strong></span>
    </div>
  `;
}

function clearMap() {
  markers.forEach((marker) => map.removeLayer(marker));
  markers = [];
  if (optimizedLine) {
    map.removeLayer(optimizedLine);
    optimizedLine = null;
  }
  if (baselineLine) {
    map.removeLayer(baselineLine);
    baselineLine = null;
  }
}

function refreshPointsUI() {
  const rows = Array.from(pointsContainer.querySelectorAll('.point-card'));
  pointsCounter.textContent = String(rows.length);
  rows.forEach((row, index) => {
    const role = row.querySelector('[data-point-role]');
    const badge = row.querySelector('[data-point-badge]');
    if (!role || !badge) return;
    let label = 'Промежуточная';
    let badgeText = String(index + 1);
    if (index === 0) {
      label = 'Старт';
      badgeText = 'S';
    } else if (index === rows.length - 1) {
      label = 'Финиш';
      badgeText = 'F';
    }
    role.textContent = label;
    badge.textContent = badgeText;
  });
  updateOptimizationHint();
  updateMapOverview({ pointsCount: rows.length });
}

function createPointRow(lat = '', lon = '', label = '') {
  const row = document.createElement('div');
  row.className = 'point-card';
  row.innerHTML = `
    <div class="point-card__top">
      <div class="point-card__role">
        <span class="point-card__role-badge" data-point-badge>1</span>
        <span data-point-role>Точка</span>
      </div>
      <button class="point-card__remove" type="button" aria-label="Удалить точку">×</button>
    </div>
    <div class="point-card__grid">
      <label class="field">
        <span>Широта</span>
        <input type="number" step="0.000001" value="${escapeHtml(lat)}" />
      </label>
      <label class="field">
        <span>Долгота</span>
        <input type="number" step="0.000001" value="${escapeHtml(lon)}" />
      </label>
      <label class="field field--full">
        <span>Метка</span>
        <input type="text" value="${escapeHtml(label)}" placeholder="Например: склад, точка доставки" />
      </label>
    </div>
  `;
  row.querySelector('.point-card__remove').addEventListener('click', () => {
    row.remove();
    refreshPointsUI();
  });
  pointsContainer.appendChild(row);
  refreshPointsUI();
}

function collectPoints() {
  return Array.from(pointsContainer.querySelectorAll('.point-card'))
    .map((row) => {
      const inputs = row.querySelectorAll('input');
      const lat = parseFloat(inputs[0].value);
      const lon = parseFloat(inputs[1].value);
      const label = inputs[2].value.trim();
      if (Number.isNaN(lat) || Number.isNaN(lon)) return null;
      return { lat, lon, label: label || null };
    })
    .filter(Boolean);
}

function syncConsumption() {
  const value = defaultConsumption[vehicleSelect.value]?.[fuelSelect.value];
  if (value) consumptionInput.value = value.toFixed(1);
}

function updateOptimizationHint() {
  if (!optimizationHintElement) return;
  if (!optimizeCheckbox.checked) {
    optimizationHintElement.textContent = 'Оптимизация отключена. Маршрут будет построен в заданном порядке.';
    return;
  }
  const pointsCount = pointsContainer.querySelectorAll('.point-card').length;
  const minPoints = fixEndsCheckbox.checked ? 4 : 3;
  if (pointsCount < minPoints) {
    optimizationHintElement.textContent = `Для перестановки нужно минимум ${minPoints} точки. Сейчас: ${pointsCount}.`;
    return;
  }
  optimizationHintElement.textContent = fixEndsCheckbox.checked
    ? `Оптимизация активна с фиксированным стартом и финишем. Текущих точек: ${pointsCount}.`
    : `Оптимизация активна с перестановкой всех точек. Текущих точек: ${pointsCount}.`;
}

function parseOptionalNumber(input, allowZero = false) {
  if (!input || input.value === '') return null;
  const value = parseFloat(input.value);
  if (Number.isNaN(value)) return null;
  if (!allowZero && value <= 0) return null;
  if (allowZero && value < 0) return null;
  return value;
}

function collectCriteriaWeights() {
  const weights = {};
  let sum = 0;
  Object.entries(weightInputs).forEach(([key, input]) => {
    const value = parseFloat(input.value || '0');
    const safeValue = Number.isNaN(value) || value < 0 ? 0 : value;
    weights[key] = safeValue;
    sum += safeValue;
  });
  return { weights, sum };
}

function collectConstraints() {
  return {
    max_distance_km: parseOptionalNumber(constraintInputs.max_distance_km),
    max_duration_min: parseOptionalNumber(constraintInputs.max_duration_min),
    max_fuel_cost: parseOptionalNumber(constraintInputs.max_fuel_cost),
    max_co2_kg: parseOptionalNumber(constraintInputs.max_co2_kg),
    max_safety_risk: parseOptionalNumber(constraintInputs.max_safety_risk, true),
  };
}

function fitToLayers(points) {
  const boundsPoints = points.map((point) => [point.lat, point.lon]);
  if (baselineLine) boundsPoints.push(...baselineLine.getLatLngs().map((point) => [point.lat, point.lng]));
  if (optimizedLine) boundsPoints.push(...optimizedLine.getLatLngs().map((point) => [point.lat, point.lng]));
  if (boundsPoints.length > 1) map.fitBounds(boundsPoints, { padding: [32, 32] });
  else if (boundsPoints.length === 1) map.setView(boundsPoints[0], 13);
}

function drawRoute(data, overridePoints = null) {
  clearMap();
  const routePoints = overridePoints || data.ordered_points || [];
  if (!routePoints.length) {
    updateMapOverview({ pointsCount: 0 });
    return;
  }
  const baselinePoints = !overridePoints ? data.comparison?.baseline_ordered_points || [] : [];
  if (baselinePoints.length > 1) {
    baselineLine = L.polyline(
      baselinePoints.map((point) => [point.lat, point.lon]),
      { color: '#7a8da6', weight: 4, opacity: 0.65, dashArray: '10 10' },
    ).addTo(map);
  }
  routePoints.forEach((point, index) => {
    const type = index === 0 ? 'start' : index === routePoints.length - 1 ? 'finish' : 'via';
    const marker = L.marker([point.lat, point.lon], {
      icon: buildMarkerIcon(type, index === 0 ? 'S' : index === routePoints.length - 1 ? 'F' : String(index + 1)),
    })
      .bindPopup(
        `<strong>${escapeHtml(point.label || `Точка ${index + 1}`)}</strong><br>${formatNumber(point.lat, 5)}, ${formatNumber(point.lon, 5)}`,
      )
      .addTo(map);
    markers.push(marker);
  });
  const polylinePoints = overridePoints
    ? routePoints.map((point) => [point.lat, point.lon])
    : (data.geometry || []).map((point) => [point[0], point[1]]);
  if (polylinePoints.length > 1) {
    optimizedLine = L.polyline(polylinePoints, {
      color: overridePoints ? '#f4b65d' : '#3fe0b8',
      weight: overridePoints ? 4 : 5,
      opacity: 0.92,
      dashArray: overridePoints ? '7 7' : null,
    }).addTo(map);
  }
  fitToLayers(routePoints);
  updateMapOverview({
    pointsCount: routePoints.length,
    status: overridePoints ? 'Просмотр альтернативы' : 'Маршрут построен',
    headline: overridePoints
      ? 'На карте показана выбранная Pareto-альтернатива.'
      : 'Оптимизированный маршрут и базовый порядок точек наложены на карту.',
    distance: data.metrics?.distance_km ?? data.total_distance_km ?? null,
    duration: data.metrics?.duration_min ?? data.total_duration_min ?? null,
  });
}

function alternativeUniqueKey(alt) {
  const metrics = alt.metrics || {};
  const points = (alt.ordered_points || [])
    .map((point) => `${Number(point.lat || 0).toFixed(6)},${Number(point.lon || 0).toFixed(6)}`)
    .join('|');
  return [
    points,
    Number(metrics.distance_km || 0).toFixed(3),
    Number(metrics.duration_min || 0).toFixed(3),
    Number(metrics.fuel_cost || 0).toFixed(3),
    Number(metrics.objective_score || 0).toFixed(5),
    metrics.feasible ? '1' : '0',
  ].join('::');
}

function getUniqueAlternatives(data) {
  const unique = [];
  const seen = new Set();
  (data.alternatives || []).forEach((alt, idx) => {
    const key = alternativeUniqueKey(alt);
    if (seen.has(key)) return;
    seen.add(key);
    unique.push({ ...alt, _sourceIndex: idx });
  });
  return unique;
}

function metricsMarkup(metrics, comparison = null) {
  if (!metrics) return '';
  const improvement = comparison?.improvement_pct || {};
  const cards = [
    ['Distance', metrics.distance_km, 'км', improvement.distance_km],
    ['Duration', metrics.duration_min, 'мин', improvement.duration_min],
    ['Fuel', metrics.fuel_liters, 'л', comparison?.improvement_pct?.fuel_cost],
    ['Fuel cost', metrics.fuel_cost, '₽', improvement.fuel_cost],
    ['CO2', metrics.co2_kg, 'кг', improvement.co2_kg],
    ['Congestion', metrics.congestion_index, 'индекс', null],
    ['Weather risk', metrics.weather_risk, 'индекс', null],
    ['Reliability', metrics.reliability_score, 'индекс', null],
    ['Safety risk', metrics.safety_risk, 'индекс', null],
    ['Tolls', metrics.toll_cost, '₽', null],
    ['Objective score', metrics.objective_score, '', comparison?.improvement_pct?.objective_score],
    ['Penalty', metrics.constraint_penalty, '', null],
  ];
  return `
    <div class="metric-grid">
      ${cards
        .map(
          ([label, value, unit, delta]) => `
            <article class="metric-card">
              <div class="metric-card__label">${label}</div>
              <div class="metric-card__value">${formatNumber(value, label === 'Objective score' ? 4 : 2)}</div>
              <div class="metric-card__unit">${unit || 'безразм.'}</div>
              <div class="metric-card__delta">${
                delta === null || delta === undefined
                  ? `Feasible: <strong>${metrics.feasible ? 'yes' : 'no'}</strong>`
                  : `Улучшение: <strong>${formatSigned(delta, 2)}%</strong>`
              }</div>
            </article>`,
        )
        .join('')}
    </div>
  `;
}

function renderHumanSummary(data, comparison) {
  const delta = comparison?.delta || {};
  const improvement = comparison?.improvement_pct || {};
  const diagnostics = data.diagnostics || {};
  const snippets = [];
  const add = (label, deltaValue, improvementValue, unit) => {
    const imp = Number(improvementValue || 0);
    const d = Number(deltaValue || 0);
    if (!Number.isFinite(imp) || !Number.isFinite(d)) return;
    if (Math.abs(imp) < 0.01 && Math.abs(d) < 1e-6) return;
    snippets.push(`${label}: ${formatSigned(imp, 2)}% (${formatSigned(-d, 3)} ${unit})`);
  };
  add('Distance', delta.distance_km, improvement.distance_km, 'км');
  add('Duration', delta.duration_min, improvement.duration_min, 'мин');
  add('Fuel cost', delta.fuel_cost, improvement.fuel_cost, '₽');
  add('CO2', delta.co2_kg, improvement.co2_kg, 'кг');
  return `
    <div class="summary-line"><strong>Итог:</strong> ${
      snippets.length ? snippets.join('; ') : 'заметных улучшений по ключевым метрикам не выявлено.'
    }</div>
    <div class="summary-line"><strong>Оценено решений:</strong> ${diagnostics.evaluated_solutions ?? '—'}. Pareto-решений: ${diagnostics.pareto_solutions ?? '—'}.</div>
  `;
}

function renderComparison(data) {
  const comparison = data.comparison;
  if (!comparison) {
    return `<div class="section-card"><h3 class="section-card__title">Сравнение до и после GA</h3><div class="section-card__subtitle">Данные сравнения недоступны.</div></div>`;
  }
  const rows = [
    ['Distance, km', comparison.baseline_metrics?.distance_km, comparison.optimized_metrics?.distance_km, comparison.delta?.distance_km, comparison.improvement_pct?.distance_km],
    ['Duration, min', comparison.baseline_metrics?.duration_min, comparison.optimized_metrics?.duration_min, comparison.delta?.duration_min, comparison.improvement_pct?.duration_min],
    ['Fuel cost, ₽', comparison.baseline_metrics?.fuel_cost, comparison.optimized_metrics?.fuel_cost, comparison.delta?.fuel_cost, comparison.improvement_pct?.fuel_cost],
    ['CO2, kg', comparison.baseline_metrics?.co2_kg, comparison.optimized_metrics?.co2_kg, comparison.delta?.co2_kg, comparison.improvement_pct?.co2_kg],
    ['Objective', comparison.baseline_metrics?.objective_score, comparison.optimized_metrics?.objective_score, comparison.delta?.objective_score, comparison.improvement_pct?.objective_score],
  ];
  return `
    <div class="section-card">
      <h3 class="section-card__title">Сравнение маршрута: Before GA vs After GA</h3>
      <div class="section-card__subtitle">Базовый порядок сравнивается с оптимизированным маршрутом.</div>
      ${renderHumanSummary(data, comparison)}
      <div class="simple-table-wrap">
        <table class="simple-table">
          <thead><tr><th>Metric</th><th>Before</th><th>After</th><th>Δ</th><th>Improvement</th></tr></thead>
          <tbody>
            ${rows
              .map(
                ([label, beforeValue, afterValue, deltaValue, improvementValue]) => `
                  <tr>
                    <td>${label}</td>
                    <td>${formatNumber(beforeValue, 3)}</td>
                    <td>${formatNumber(afterValue, 3)}</td>
                    <td>${formatSigned(deltaValue, 3)}</td>
                    <td>${formatSigned(improvementValue, 2)}%</td>
                  </tr>`,
              )
              .join('')}
          </tbody>
        </table>
      </div>
      <div class="comparison-grid">
        <div class="section-card">
          <h3 class="section-card__title">Порядок до оптимизации</h3>
          <div class="route-order">
            ${(comparison.baseline_ordered_points || [])
              .map(
                (point, index) => `
                  <div class="route-order__row">
                    <span class="route-order__index">${index + 1}</span>
                    <span>${escapeHtml(point.label || `Точка ${index + 1}`)}</span>
                  </div>`,
              )
              .join('')}
          </div>
        </div>
        <div class="section-card">
          <h3 class="section-card__title">Порядок после оптимизации</h3>
          <div class="route-order">
            ${(data.ordered_points || [])
              .map(
                (point, index) => `
                  <div class="route-order__row">
                    <span class="route-order__index">${index + 1}</span>
                    <span>${escapeHtml(point.label || `Точка ${index + 1}`)}</span>
                  </div>`,
              )
              .join('')}
          </div>
        </div>
      </div>
    </div>`;
}

function renderWeights(data) {
  if (!data.dynamic_weights) return '<div class="summary-line">Динамические веса не применялись.</div>';
  const dw = data.dynamic_weights;
  const criteria = [['distance', 'Distance'], ['duration', 'Duration'], ['fuel_cost', 'Fuel cost'], ['emissions', 'Emissions'], ['congestion', 'Congestion'], ['weather_risk', 'Weather risk'], ['reliability', 'Reliability'], ['safety', 'Safety'], ['tolls', 'Tolls']];
  return `
    <div class="summary-line"><strong>Triggers:</strong> ${(dw.triggers || []).length ? dw.triggers.map((item) => escapeHtml(item)).join(', ') : 'none'}</div>
    <div class="simple-table-wrap">
      <table class="simple-table">
        <thead><tr><th>Criterion</th><th>Base</th><th>Adjusted</th><th>Δ</th></tr></thead>
        <tbody>
          ${criteria
            .map(([key, label]) => {
              const base = dw.base?.[key] ?? 0;
              const adjusted = dw.adjusted?.[key] ?? 0;
              return `<tr><td>${label}</td><td>${formatNumber(base, 3)}</td><td>${formatNumber(adjusted, 3)}</td><td>${formatSigned(adjusted - base, 3)}</td></tr>`;
            })
            .join('')}
        </tbody>
      </table>
    </div>`;
}

function renderScoreExplanation(title, explanation) {
  if (!explanation || !(explanation.components || []).length) {
    return `<div class="section-card"><h3 class="section-card__title">${title}</h3><div class="section-card__subtitle">Детализация score недоступна.</div></div>`;
  }
  return `
    <div class="section-card">
      <h3 class="section-card__title">${title}</h3>
      <div class="section-card__subtitle">Mode: ${escapeHtml(explanation.score_mode)} · Total: ${formatNumber(explanation.total_score, 4)}</div>
      <div class="simple-table-wrap">
        <table class="simple-table">
          <thead><tr><th>Component</th><th>Weight</th><th>Raw</th><th>Norm</th><th>Contribution</th></tr></thead>
          <tbody>
            ${(explanation.components || [])
              .map(
                (item) => `<tr><td>${escapeHtml(item.label || item.key)}</td><td>${formatNumber(item.weight, 3)}</td><td>${formatNumber(item.raw_value, 3)}</td><td>${formatNumber(item.normalized_value, 3)}</td><td>${formatNumber(item.contribution, 3)}</td></tr>`,
              )
              .join('')}
          </tbody>
        </table>
      </div>
    </div>`;
}

function getAlternativeBadges(alternatives, alt) {
  if (!alternatives.length) return [];
  const sorted = (selector) => [...alternatives].sort((a, b) => selector(a) - selector(b))[0];
  const badges = [];
  if (sorted((item) => item.metrics?.objective_score ?? Infinity) === alt) badges.push('Лучший');
  if (sorted((item) => item.metrics?.duration_min ?? Infinity) === alt) badges.push('Быстрее');
  if (sorted((item) => item.metrics?.fuel_cost ?? Infinity) === alt) badges.push('Дешевле');
  if (sorted((item) => item.metrics?.safety_risk ?? Infinity) === alt) badges.push('Безопаснее');
  return badges;
}

function renderAlternativeCards(data, limit = null) {
  const alternatives = getUniqueAlternatives(data);
  if (!alternatives.length) {
    return '<div class="empty-state"><div class="empty-state__title">Pareto-альтернативы отсутствуют</div><div class="empty-state__text">Для этого запуска дополнительные альтернативы не сформировались.</div></div>';
  }
  return `
    <div class="alt-grid">
      ${(limit ? alternatives.slice(0, limit) : alternatives)
        .map((alt) => {
          const badges = getAlternativeBadges(alternatives, alt);
          return `
            <article class="alt-card">
              <div class="alt-card__top">
                <div>
                  <div class="surface__eyebrow">Alternative #${alt.rank || alt._sourceIndex + 1}</div>
                  <h3 class="section-card__title">Оценка ${formatNumber(alt.metrics?.objective_score, 4)}</h3>
                  <div class="alt-card__meta">
                    ${badges.map((badge) => `<span class="pill pill--highlight">${badge}</span>`).join('')}
                    <span class="pill ${alt.metrics?.feasible ? 'pill--soft' : 'pill--danger'}">${alt.metrics?.feasible ? 'Feasible' : 'Not feasible'}</span>
                  </div>
                </div>
                <button class="btn btn--inline" type="button" data-alt-index="${alt._sourceIndex}">Показать на карте</button>
              </div>
              <div class="alt-card__stats">
                <div class="data-item"><div class="data-item__label">Distance</div><div class="data-item__value">${formatNumber(alt.metrics?.distance_km, 1)} км</div></div>
                <div class="data-item"><div class="data-item__label">Duration</div><div class="data-item__value">${formatNumber(alt.metrics?.duration_min, 0)} мин</div></div>
                <div class="data-item"><div class="data-item__label">Fuel cost</div><div class="data-item__value">${formatNumber(alt.metrics?.fuel_cost, 0)} ₽</div></div>
                <div class="data-item"><div class="data-item__label">Safety risk</div><div class="data-item__value">${formatNumber(alt.metrics?.safety_risk, 3)}</div></div>
              </div>
              <div class="alt-order">
                ${(alt.ordered_points || [])
                  .map((point, index) => `<span class="alt-order__item">${index + 1}. ${escapeHtml(point.label || `Точка ${index + 1}`)}</span>`)
                  .join('')}
              </div>
            </article>`;
        })
        .join('')}
    </div>`;
}

function renderFuelBlock(data) {
  const fuel = data.fuel_cost;
  if (!fuel) return `<div class="section-card"><h3 class="section-card__title">Топливо</h3><div class="section-card__subtitle">Топливный блок отсутствует в ответе.</div></div>`;
  const sourceText = fuel.price_source_url
    ? `<a href="${escapeHtml(fuel.price_source_url)}" target="_blank" rel="noopener">${escapeHtml(fuel.price_source)}</a>`
    : escapeHtml(fuel.price_source);
  return `
    <div class="section-card">
      <h3 class="section-card__title">Топливный блок</h3>
      <div class="data-grid">
        <div class="data-item"><div class="data-item__label">Объём</div><div class="data-item__value">${formatNumber(fuel.liters_total, 2)} л</div></div>
        <div class="data-item"><div class="data-item__label">Цена за литр</div><div class="data-item__value">${formatNumber(fuel.price_per_liter, 2)} ${escapeHtml(fuel.currency)}/л</div></div>
        <div class="data-item"><div class="data-item__label">Итоговая стоимость</div><div class="data-item__value">${formatNumber(fuel.total_cost, 2)} ${escapeHtml(fuel.currency)}</div></div>
        <div class="data-item"><div class="data-item__label">Рельеф</div><div class="data-item__value">×${formatNumber(fuel.terrain_multiplier, 3)}</div></div>
        <div class="data-item"><div class="data-item__label">Горы</div><div class="data-item__value">×${formatNumber(fuel.mountain_multiplier, 3)}</div></div>
        <div class="data-item"><div class="data-item__label">Температура</div><div class="data-item__value">×${formatNumber(fuel.temperature_multiplier, 3)}</div></div>
      </div>
      <div class="summary-line"><strong>Источник цены:</strong> ${sourceText} ${fuel.price_date ? `· ${escapeHtml(fuel.price_date)}` : ''}</div>
    </div>`;
}

function renderSourcesBlock(data) {
  const sources = data.data_sources;
  if (!sources) return `<div class="section-card"><h3 class="section-card__title">Источники данных</h3><div class="section-card__subtitle">Список источников недоступен.</div></div>`;
  const items = [['Routing', sources.routing], ['Matrix', sources.matrix], ['Weather', sources.weather], ['Elevation', sources.elevation], ['Traffic', sources.traffic], ['Tolls', sources.tolls], ['Fuel prices', sources.fuel_prices]];
  return `
    <div class="section-card">
      <h3 class="section-card__title">Источники данных</h3>
      <div class="simple-grid--small">
        ${items.map(([label, value]) => `<div class="data-item"><div class="data-item__label">${label}</div><div class="data-item__value">${escapeHtml(value)}</div></div>`).join('')}
      </div>
    </div>`;
}

function renderDiagnosticsBlock(data) {
  const diagnostics = data.diagnostics;
  if (!diagnostics) return `<div class="section-card"><h3 class="section-card__title">Диагностика оптимизатора</h3><div class="section-card__subtitle">Диагностика для этого запуска отсутствует.</div></div>`;
  const items = [
    ['Mode', diagnostics.mode],
    ['Optimization active', diagnostics.optimization_active ? 'yes' : 'no'],
    ['Reason', diagnostics.optimization_reason || '—'],
    ['Score mode', diagnostics.score_mode || '—'],
    ['Generations', diagnostics.generations],
    ['Population', diagnostics.population_size],
    ['Crossover', formatNumber(diagnostics.crossover_rate, 2)],
    ['Mutation', formatNumber(diagnostics.mutation_rate, 2)],
    ['Stagnation', diagnostics.stagnation_generations],
    ['Evaluated', diagnostics.evaluated_solutions],
    ['Pareto', diagnostics.pareto_solutions],
  ];
  return `
    <div class="section-card">
      <h3 class="section-card__title">Диагностика оптимизатора</h3>
      <div class="simple-grid--small">
        ${items.map(([label, value]) => `<div class="data-item"><div class="data-item__label">${label}</div><div class="data-item__value">${escapeHtml(value)}</div></div>`).join('')}
      </div>
    </div>`;
}

function renderResult(data, meta = {}) {
  currentData = data;
  const routeOrder = (data.ordered_points || []).map((point, index) => escapeHtml(point.label || `Точка ${index + 1}`)).join(' → ');
  const uniqueAlternatives = getUniqueAlternatives(data);
  const createdAt = meta.createdAt ? formatDateTime(meta.createdAt) : null;
  resultContainer.innerHTML = `
    <div class="result-layout">
      <section class="result-banner">
        <div>
          <div class="surface__eyebrow">${meta.fromHistory ? 'Recovered Run' : 'Optimization Complete'}</div>
          <h3 class="surface__title">Маршрут рассчитан и готов к анализу</h3>
          <div class="section-card__subtitle">Порядок точек: ${routeOrder || '—'}</div>
          <div class="result-banner__meta">
            <span class="chip"><strong>Run:</strong> ${escapeHtml(data.run_id || '—')}</span>
            <span class="chip"><strong>Provider:</strong> ${escapeHtml(data.provider || '—')}</span>
            ${createdAt ? `<span class="chip"><strong>Создан:</strong> ${escapeHtml(createdAt)}</span>` : ''}
            <span class="chip"><strong>Pareto:</strong> ${uniqueAlternatives.length}</span>
          </div>
        </div>
        <div class="result-actions">
          ${uniqueAlternatives.length ? '<button class="btn btn--inline" type="button" id="open-pareto">Все Pareto-альтернативы</button>' : ''}
          ${data.run_id ? `<a class="btn btn--inline btn--link" href="/api/runs/${escapeHtml(data.run_id)}/report.csv" target="_blank" rel="noopener">Экспорт CSV</a>` : ''}
        </div>
      </section>
      ${metricsMarkup(data.metrics, data.comparison)}
      ${renderComparison(data)}
      ${renderScoreExplanation('Score decomposition: Before GA', data.comparison?.baseline_score)}
      ${renderScoreExplanation('Score decomposition: After GA', data.comparison?.optimized_score)}
      ${renderFuelBlock(data)}
      <section class="section-card"><h3 class="section-card__title">Динамические веса</h3>${renderWeights(data)}</section>
      ${renderSourcesBlock(data)}
      ${renderDiagnosticsBlock(data)}
      <section class="section-card">
        <h3 class="section-card__title">Pareto-альтернативы</h3>
        <div class="section-card__subtitle">Быстрый просмотр уникальных альтернатив; полная таблица доступна в модальном окне.</div>
        ${renderAlternativeCards(data, 3)}
      </section>
    </div>`;
  const openParetoButton = document.getElementById('open-pareto');
  if (openParetoButton) openParetoButton.addEventListener('click', renderParetoModal);
}

function renderParetoModal() {
  if (!currentData) return;
  paretoModalBody.innerHTML = renderAlternativeCards(currentData);
  openModal('pareto-modal');
}

function showInlineMessage(title, text) {
  resultContainer.innerHTML = `<div class="empty-state"><div class="empty-state__title">${escapeHtml(title)}</div><div class="empty-state__text">${escapeHtml(text)}</div></div>`;
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const contentType = response.headers.get('content-type') || '';
  const payload = contentType.includes('application/json') ? await response.json() : await response.text();
  if (!response.ok) {
    throw new Error(typeof payload === 'string' ? payload : payload?.detail || 'Сервер вернул ошибку.');
  }
  return payload;
}

async function buildRoute() {
  const points = collectPoints();
  if (points.length < 2) return showInlineMessage('Недостаточно точек', 'Нужно указать минимум две точки маршрута.');
  const { weights, sum } = collectCriteriaWeights();
  if (sum <= 0) return showInlineMessage('Некорректные веса', 'Сумма весов критериев должна быть больше нуля.');
  const payload = {
    points,
    optimize: optimizeCheckbox.checked,
    fix_ends: fixEndsCheckbox.checked,
    profile: profileSelect.value,
    vehicle_class: vehicleSelect.value,
    fuel_type: fuelSelect.value,
    fuel_consumption_l_per_100km: parseFloat(consumptionInput.value),
    optimize_mode: optimizeModeInput.value,
    priority_profile: priorityProfileInput.value,
    criteria_weights: weights,
    constraints: collectConstraints(),
    use_dynamic_weights: useDynamicWeightsInput.checked,
    departure_at: departureInput.value ? new Date(departureInput.value).toISOString() : null,
    population_size: parseInt(populationInput.value, 10),
    generations: parseInt(generationsInput.value, 10),
    crossover_rate: parseFloat(crossoverInput.value),
    mutation_rate: parseFloat(mutationInput.value),
    max_alternatives: parseInt(maxAlternativesInput.value, 10),
  };
  if (!Number.isFinite(payload.fuel_consumption_l_per_100km) || payload.fuel_consumption_l_per_100km <= 0) {
    return showInlineMessage('Некорректный расход топлива', 'Укажите положительное значение расхода топлива.');
  }
  try {
    showLoading('Система строит маршрут, рассчитывает метрики и формирует Pareto-альтернативы.');
    const data = await fetchJson('/api/routes', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    renderResult(data);
    drawRoute(data);
    await fetchHistory();
    switchView('optimization');
  } catch (error) {
    showInlineMessage('Ошибка при построении маршрута', error.message || 'Не удалось выполнить расчёт.');
  } finally {
    hideLoading();
  }
}

function applyRequestToForm(request) {
  pointsContainer.innerHTML = '';
  (request.points || []).forEach((point) => createPointRow(point.lat, point.lon, point.label || ''));
  optimizeCheckbox.checked = Boolean(request.optimize);
  fixEndsCheckbox.checked = Boolean(request.fix_ends);
  profileSelect.value = request.profile || 'driving';
  vehicleSelect.value = request.vehicle_class || 'passenger';
  fuelSelect.value = request.fuel_type || 'petrol';
  consumptionInput.value = request.fuel_consumption_l_per_100km ?? '';
  optimizeModeInput.value = request.optimize_mode || 'weighted';
  priorityProfileInput.value = request.priority_profile || 'balanced';
  useDynamicWeightsInput.checked = request.use_dynamic_weights !== false;
  const departureDate = request.departure_at ? new Date(request.departure_at) : null;
  departureInput.value = departureDate && !Number.isNaN(departureDate.getTime())
    ? `${departureDate.getFullYear()}-${String(departureDate.getMonth() + 1).padStart(2, '0')}-${String(departureDate.getDate()).padStart(2, '0')}T${String(departureDate.getHours()).padStart(2, '0')}:${String(departureDate.getMinutes()).padStart(2, '0')}`
    : '';
  populationInput.value = request.population_size ?? 96;
  generationsInput.value = request.generations ?? 120;
  crossoverInput.value = request.crossover_rate ?? 0.88;
  mutationInput.value = request.mutation_rate ?? 0.22;
  maxAlternativesInput.value = request.max_alternatives ?? 8;
  Object.entries(weightInputs).forEach(([key, input]) => {
    input.value = request.criteria_weights?.[key] ?? input.value;
  });
  Object.entries(constraintInputs).forEach(([key, input]) => {
    const value = request.constraints?.[key];
    input.value = value === null || value === undefined ? '' : value;
  });
  refreshPointsUI();
}

async function loadRun(runId) {
  try {
    showLoading('Загружаю сохранённый запуск и восстанавливаю состояние интерфейса.');
    const details = await fetchJson(`/api/runs/${encodeURIComponent(runId)}`);
    applyRequestToForm(details.request || {});
    renderResult(details.response || {}, { fromHistory: true, createdAt: details.created_at });
    drawRoute(details.response || {});
    switchView('optimization');
  } catch (error) {
    showInlineMessage('Не удалось загрузить запуск', error.message || 'Ошибка при чтении истории.');
  } finally {
    hideLoading();
  }
}

function renderHistory(list) {
  if (!Array.isArray(list) || !list.length) {
    historyStats.innerHTML = '';
    historyTable.innerHTML = `<div class="empty-state"><div class="empty-state__title">История пока пуста</div><div class="empty-state__text">После первого успешного запуска здесь появится журнал расчётов.</div></div>`;
    return;
  }
  const avgScore = list.reduce((sum, item) => sum + Number(item.objective_score || 0), 0) / list.length;
  const feasibleTotal = list.reduce((sum, item) => sum + Number(item.feasible_count || 0), 0);
  historyStats.innerHTML = `
    <article class="metric-card"><div class="metric-card__label">Всего запусков</div><div class="metric-card__value">${list.length}</div><div class="metric-card__unit">записей</div></article>
    <article class="metric-card"><div class="metric-card__label">Средняя objective score</div><div class="metric-card__value">${formatNumber(avgScore, 3)}</div><div class="metric-card__unit">по истории</div></article>
    <article class="metric-card"><div class="metric-card__label">Feasible alternatives</div><div class="metric-card__value">${feasibleTotal}</div><div class="metric-card__unit">накоплено</div></article>
    <article class="metric-card"><div class="metric-card__label">Последний провайдер</div><div class="metric-card__value">${escapeHtml(list[0].provider_summary || '—')}</div><div class="metric-card__unit">${formatDateShort(list[0].created_at)}</div></article>`;
  historyTable.innerHTML = list
    .map(
      (item) => `
        <article class="history-row">
          <div class="history-row__top">
            <div>
              <div class="surface__eyebrow">Run ${escapeHtml(item.run_id)}</div>
              <h3 class="section-card__title">${escapeHtml(item.provider_summary || 'Маршрутный запуск')}</h3>
              <div class="section-card__subtitle">Создан: ${formatDateTime(item.created_at)}</div>
            </div>
            <div class="result-actions">
              <button class="btn btn--inline" type="button" data-history-open="${escapeHtml(item.run_id)}">Открыть</button>
              <a class="btn btn--inline btn--link" href="/api/runs/${escapeHtml(item.run_id)}/report.csv" target="_blank" rel="noopener">CSV</a>
            </div>
          </div>
          <div class="history-row__details">
            <div class="data-item"><div class="data-item__label">Run ID</div><div class="data-item__value mono">${escapeHtml(item.run_id)}</div></div>
            <div class="data-item"><div class="data-item__label">Objective</div><div class="data-item__value">${formatNumber(item.objective_score, 4)}</div></div>
            <div class="data-item"><div class="data-item__label">Feasible</div><div class="data-item__value">${escapeHtml(item.feasible_count)}</div></div>
            <div class="data-item"><div class="data-item__label">Дата</div><div class="data-item__value">${formatDateShort(item.created_at)}</div></div>
          </div>
        </article>`,
    )
    .join('');
}

async function fetchHistory() {
  try {
    renderHistory(await fetchJson('/api/runs?limit=20'));
  } catch (error) {
    historyTable.innerHTML = `<div class="empty-state"><div class="empty-state__title">Не удалось загрузить историю</div><div class="empty-state__text">${escapeHtml(error.message || 'Ошибка при обращении к серверу.')}</div></div>`;
  }
}

document.addEventListener('click', (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return;
  const modalId = target.getAttribute('data-modal-close');
  if (modalId) closeModal(modalId);
});

navButtons.forEach((button) => {
  button.addEventListener('click', () => switchView(button.dataset.viewTarget));
});

addButton.addEventListener('click', () => createPointRow());
clearButton.addEventListener('click', () => {
  pointsContainer.innerHTML = '';
  clearMap();
  currentData = null;
  refreshPointsUI();
  showInlineMessage('Маршрут очищен', 'Добавьте новые точки маршрута или выберите сохранённый запуск из истории.');
});
buildButton.addEventListener('click', buildRoute);
vehicleSelect.addEventListener('change', syncConsumption);
fuelSelect.addEventListener('change', syncConsumption);
optimizeCheckbox.addEventListener('change', updateOptimizationHint);
fixEndsCheckbox.addEventListener('change', updateOptimizationHint);
resultContainer.addEventListener('click', (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement) || currentData === null) return;
  const altIndex = target.getAttribute('data-alt-index');
  if (altIndex === null) return;
  const alt = (currentData.alternatives || [])[parseInt(altIndex, 10)];
  if (alt) drawRoute(currentData, alt.ordered_points || []);
});
paretoModalBody.addEventListener('click', (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement) || currentData === null) return;
  const altIndex = target.getAttribute('data-alt-index');
  if (altIndex === null) return;
  const alt = (currentData.alternatives || [])[parseInt(altIndex, 10)];
  if (!alt) return;
  drawRoute(currentData, alt.ordered_points || []);
  closeModal('pareto-modal');
});
historyTable.addEventListener('click', (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return;
  const runId = target.getAttribute('data-history-open');
  if (runId) loadRun(runId);
});
refreshHistoryButton.addEventListener('click', fetchHistory);
historyShortcutButton.addEventListener('click', () => switchView('history'));

document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') closeModal('pareto-modal');
});

map.on('click', (event) => {
  createPointRow(event.latlng.lat.toFixed(6), event.latlng.lng.toFixed(6));
});

if (departureInput) {
  const now = new Date();
  departureInput.value = new Date(now.getTime() - now.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
}

createPointRow(51.672, 39.1843, 'Старт');
createPointRow(51.66, 39.2, 'Финиш');
syncConsumption();
refreshPointsUI();
fetchHistory();
