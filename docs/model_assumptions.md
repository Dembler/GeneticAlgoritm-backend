# Model Assumptions and Calibration Scope

This project intentionally combines strict formulas with practical heuristics.

## Implemented as close-form formulas
- Multi-objective optimization core based on NSGA-II components:
  - non-dominated sorting
  - crowding distance
  - crossover and mutation over route permutations
- Mountain fuel correction uses calibrated slope and altitude multipliers.
- Temperature correction uses DOE/FuelEconomy-inspired cold and hot weather anchors.

## Implemented as engineering heuristics
- Dynamic criteria weights are rule-based multipliers over normalized base weights.
- Priority profiles (`fastest`, `cheapest`, `safest`, `greenest`) are explicit weighting rules.
- Traffic/tolls/weather fallback behavior is provider-dependent and can degrade to safe defaults.

## Known scope boundaries
- Current traffic source may be disabled (`traffic-disabled`), so congestion dynamics can be limited.
- Dynamic weighting is not yet a learned model (no logistic-regression/ML estimator in runtime path).
- GA hyperparameters are currently static inputs, not adaptive schedules.

## Interpretation policy for source papers
- PDF sources are used as calibration references and motivation.
- Runtime formulas remain robust for mixed route scenarios and may differ from paper-specific experimental setups.
