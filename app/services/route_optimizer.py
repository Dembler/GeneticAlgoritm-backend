from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import logging
import os
import random
from time import perf_counter

from app.domain.models import (
    CriteriaWeights,
    OptimizationDiagnostics,
    OptimizationMode,
    OptimizationReason,
    RouteMetrics,
    RouteRequest,
    ScoreMode,
)
from app.services.context_service import OptimizationContext
from app.services.criteria_service import CandidateEvaluation, CriteriaService
from app.services.fuel_cost import FuelCostService, FuelPriceSnapshot

logger = logging.getLogger(__name__)

_WORKER_REQUEST: RouteRequest | None = None
_WORKER_CONTEXT: OptimizationContext | None = None
_WORKER_FUEL_PRICES: FuelPriceSnapshot | None = None
_WORKER_CRITERIA_SERVICE: CriteriaService | None = None


class _WorkerFuelPriceService:
    async def get_prices(self) -> FuelPriceSnapshot:
        raise RuntimeError("GA worker does not fetch fuel prices")


def _decode_genome(genome: tuple[int, ...], fix_ends: bool, points_count: int) -> list[int]:
    middle = list(genome)
    if fix_ends:
        return [0] + middle + [points_count - 1]
    return middle


def _init_evaluation_worker(
    request: RouteRequest,
    context: OptimizationContext,
    fuel_prices: FuelPriceSnapshot,
) -> None:
    global _WORKER_REQUEST, _WORKER_CONTEXT, _WORKER_FUEL_PRICES, _WORKER_CRITERIA_SERVICE
    _WORKER_REQUEST = request
    _WORKER_CONTEXT = context
    _WORKER_FUEL_PRICES = fuel_prices
    _WORKER_CRITERIA_SERVICE = CriteriaService(FuelCostService(_WorkerFuelPriceService()))


def _evaluate_genome_in_worker(
    task: tuple[tuple[int, ...], bool, int],
) -> tuple[tuple[int, ...], CandidateEvaluation]:
    if _WORKER_REQUEST is None or _WORKER_CONTEXT is None or _WORKER_FUEL_PRICES is None:
        raise RuntimeError("GA worker is not initialized")
    if _WORKER_CRITERIA_SERVICE is None:
        raise RuntimeError("GA worker criteria service is not initialized")
    genome, fix_ends, points_count = task
    order = _decode_genome(genome, fix_ends, points_count)
    return genome, _WORKER_CRITERIA_SERVICE.evaluate(order, _WORKER_REQUEST, _WORKER_CONTEXT, _WORKER_FUEL_PRICES)


@dataclass
class CandidateState:
    genome: tuple[int, ...]
    evaluation: CandidateEvaluation
    rank: int = 0
    crowding: float = 0.0


@dataclass
class ParetoItem:
    evaluation: CandidateEvaluation
    rank: int
    crowding: float


@dataclass
class FinalSelection:
    state: CandidateState
    baseline_evaluation: CandidateEvaluation
    baseline_guard_applied: bool
    original_best_score: float
    baseline_score: float
    baseline_guard_reason: str | None
    final_selected_from: str
    final_selection_reason: str


@dataclass
class OptimizationResult:
    best: CandidateEvaluation
    pareto: list[ParetoItem]
    diagnostics: OptimizationDiagnostics
    matrix_provider: str
    population_memory_orders: list[list[int]]


class RouteOptimizer:
    _NON_REGRESSION_EPSILON = 1e-9

    def __init__(
        self,
        criteria_service: CriteriaService,
        parallel_workers: int | None = None,
        parallel_min_batch_size: int = 48,
    ) -> None:
        self._criteria_service = criteria_service
        self._parallel_workers = parallel_workers
        self._parallel_min_batch_size = max(1, parallel_min_batch_size)

    def evaluate_order(
        self,
        request: RouteRequest,
        context: OptimizationContext,
        order_indices: list[int],
        weights: CriteriaWeights,
        fuel_prices: FuelPriceSnapshot,
    ) -> CandidateEvaluation:
        evaluation = self._criteria_service.evaluate(order_indices, request, context, fuel_prices)
        self._criteria_service.assign_weighted_scores([evaluation], weights)
        return evaluation

    def optimize(
        self,
        request: RouteRequest,
        context: OptimizationContext,
        weights: CriteriaWeights,
        fuel_prices: FuelPriceSnapshot,
    ) -> OptimizationResult:
        points_count = len(context.points)
        if not request.optimize:
            order = list(range(points_count))
            evaluation = self._criteria_service.evaluate(order, request, context, fuel_prices)
            self._criteria_service.assign_weighted_scores([evaluation], weights)
            return self._single_candidate_result(
                request=request,
                context=context,
                evaluation=evaluation,
                optimization_active=False,
                optimization_reason=OptimizationReason.optimize_disabled,
                score_mode=ScoreMode.absolute_single_candidate,
            )

        if points_count <= 2:
            evaluation = self._criteria_service.evaluate(list(range(points_count)), request, context, fuel_prices)
            self._criteria_service.assign_weighted_scores([evaluation], weights)
            return self._single_candidate_result(
                request=request,
                context=context,
                evaluation=evaluation,
                optimization_active=False,
                optimization_reason=OptimizationReason.not_enough_points,
                score_mode=ScoreMode.absolute_single_candidate,
            )

        rng = random.Random(request.random_seed)
        fix_ends = request.fix_ends
        use_pareto = request.optimize_mode == OptimizationMode.pareto
        middle_indices = list(range(1, points_count - 1)) if fix_ends else list(range(points_count))
        genome_len = len(middle_indices)
        forbidden_edges = self._forbidden_edges(request, context)
        repaired_solutions = 0
        unrepairable_candidates = 0
        if genome_len <= 1:
            order = self._decode(tuple(middle_indices), fix_ends, points_count)
            evaluation = self._criteria_service.evaluate(order, request, context, fuel_prices)
            self._criteria_service.assign_weighted_scores([evaluation], weights)
            return self._single_candidate_result(
                request=request,
                context=context,
                evaluation=evaluation,
                optimization_active=False,
                optimization_reason=OptimizationReason.fixed_route,
                score_mode=ScoreMode.absolute_single_candidate,
                forbidden_edges=len(forbidden_edges),
            )

        population_size = max(20, request.population_size)
        generations = max(20, request.generations)
        eval_cache: dict[tuple[int, ...], CandidateEvaluation] = {}
        evaluated_solutions = 0
        evaluation_ms = 0.0
        parallel_batches = 0
        parallel_evaluations = 0
        optimizer_start = perf_counter()
        worker_count = self._parallel_worker_count(population_size)
        process_pool: ProcessPoolExecutor | None = None
        if worker_count > 1:
            try:
                process_pool = ProcessPoolExecutor(
                    max_workers=worker_count,
                    initializer=_init_evaluation_worker,
                    initargs=(request, context, fuel_prices),
                )
            except Exception:
                logger.exception("GA parallel worker pool initialization failed; using sequential evaluation")
                process_pool = None
                worker_count = 1
        logger.warning(
            "GA optimization started: points=%d genome_len=%d population_size=%d generations=%d mode=%s parallel_workers=%d parallel_min_batch_size=%d",
            points_count,
            genome_len,
            population_size,
            generations,
            getattr(request.optimize_mode, "value", request.optimize_mode),
            worker_count,
            self._parallel_min_batch_size,
        )

        def evaluate_missing_genomes(genomes: list[tuple[int, ...]]) -> None:
            nonlocal evaluated_solutions, evaluation_ms, parallel_batches, parallel_evaluations
            nonlocal process_pool, worker_count
            missing: list[tuple[int, ...]] = []
            seen_missing: set[tuple[int, ...]] = set()
            for genome in genomes:
                if genome in eval_cache or genome in seen_missing:
                    continue
                missing.append(genome)
                seen_missing.add(genome)
            if not missing:
                return

            batch_start = perf_counter()
            if process_pool is not None and len(missing) >= self._parallel_min_batch_size:
                try:
                    tasks = [(genome, fix_ends, points_count) for genome in missing]
                    chunksize = self._parallel_chunksize(len(tasks), worker_count)
                    for genome, evaluation in process_pool.map(
                        _evaluate_genome_in_worker,
                        tasks,
                        chunksize=chunksize,
                    ):
                        eval_cache[genome] = evaluation
                    evaluated_solutions += len(missing)
                    parallel_batches += 1
                    parallel_evaluations += len(missing)
                    evaluation_ms += (perf_counter() - batch_start) * 1000.0
                    return
                except Exception:
                    logger.exception("GA parallel batch evaluation failed; switching to sequential evaluation")
                    process_pool.shutdown(cancel_futures=True)
                    process_pool = None
                    worker_count = 1

            for genome in missing:
                order = self._decode(genome, fix_ends, points_count)
                eval_cache[genome] = self._criteria_service.evaluate(order, request, context, fuel_prices)
            evaluated_solutions += len(missing)
            evaluation_ms += (perf_counter() - batch_start) * 1000.0

        def evaluate_genome(genome: tuple[int, ...]) -> CandidateEvaluation:
            cached = eval_cache.get(genome)
            if cached is not None:
                return cached
            evaluate_missing_genomes([genome])
            return eval_cache[genome]

        try:
            population_genomes = self._initial_population(
                middle_indices=middle_indices,
                population_size=population_size,
                rng=rng,
                context=context,
                fix_ends=fix_ends,
                points_count=points_count,
                warm_start_orders=request.warm_start_orders,
                forbidden_edges=forbidden_edges,
            )
            elite_genomes = set(population_genomes[: min(2, len(population_genomes))])

            def repair_genome(genome: tuple[int, ...]) -> tuple[int, ...]:
                nonlocal repaired_solutions, unrepairable_candidates
                repaired = self._repair_genome(genome, fix_ends, points_count, forbidden_edges)
                if repaired != genome:
                    repaired_solutions += 1
                if self._bad_edge_count(self._decode(repaired, fix_ends, points_count), forbidden_edges) > 0:
                    unrepairable_candidates += 1
                return repaired

            evaluate_missing_genomes(population_genomes)
            states = [CandidateState(genome=g, evaluation=evaluate_genome(g)) for g in population_genomes]
            self._criteria_service.assign_weighted_scores([s.evaluation for s in states], weights)
            if use_pareto:
                self._rank_and_crowding(states)

            best_score = min([s.evaluation.metrics.objective_score for s in states]) if states else float("inf")
            stagnation_generations = 0

            for _generation in range(generations):
                offspring_genomes: list[tuple[int, ...]] = []
                while len(offspring_genomes) < population_size:
                    if use_pareto:
                        p1 = self._tournament_select(states, rng)
                        p2 = self._tournament_select(states, rng)
                    else:
                        p1 = self._tournament_select_weighted(states, rng)
                        p2 = self._tournament_select_weighted(states, rng)
                    child1, child2 = p1.genome, p2.genome
                    if rng.random() <= request.crossover_rate:
                        child1, child2 = self._pmx_crossover(p1.genome, p2.genome, rng)
                    child1 = self._mutate(child1, rng, request.mutation_rate)
                    child2 = self._mutate(child2, rng, request.mutation_rate)
                    child1 = repair_genome(child1)
                    child2 = repair_genome(child2)
                    offspring_genomes.append(child1)
                    if len(offspring_genomes) < population_size:
                        offspring_genomes.append(child2)

                evaluate_missing_genomes(offspring_genomes)
                offspring_states = [CandidateState(genome=g, evaluation=evaluate_genome(g)) for g in offspring_genomes]
                merged = states + offspring_states
                self._criteria_service.assign_weighted_scores([s.evaluation for s in merged], weights)
                if use_pareto:
                    merged_fronts = self._rank_and_crowding(merged)
                    states = self._select_next_generation(merged_fronts, population_size)
                else:
                    states = self._select_next_generation_weighted(merged, population_size)
                states = self._preserve_elites(
                    selected=states,
                    elite_genomes=elite_genomes,
                    evaluate_genome=evaluate_genome,
                    weights=weights,
                    population_size=population_size,
                    use_pareto=use_pareto,
                )

                generation_best = min([s.evaluation.metrics.objective_score for s in states]) if states else float("inf")
                if generation_best + 1e-8 < best_score:
                    best_score = generation_best
                    stagnation_generations = 0
                else:
                    stagnation_generations += 1

                if use_pareto:
                    self._rank_and_crowding(states)
        finally:
            if process_pool is not None:
                process_pool.shutdown(cancel_futures=True)

        if not states:
            fallback_eval = self._criteria_service.evaluate(list(range(points_count)), request, context, fuel_prices)
            self._criteria_service.assign_weighted_scores([fallback_eval], weights)
            diagnostics = OptimizationDiagnostics(
                mode=request.optimize_mode,
                optimization_active=True,
                optimization_reason=None,
                score_mode=ScoreMode.absolute_single_candidate,
                generations=1,
                population_size=1,
                crossover_rate=request.crossover_rate,
                mutation_rate=request.mutation_rate,
                stagnation_generations=0,
                evaluated_solutions=evaluated_solutions,
                pareto_solutions=1,
                warm_start_solutions=len(request.warm_start_orders),
                population_memory_solutions=1,
                repaired_solutions=repaired_solutions,
                forbidden_edges=len(forbidden_edges),
                forbidden_edges_count=len(forbidden_edges),
                unrepairable_candidates=unrepairable_candidates,
                infrastructure_violations_count=len(forbidden_edges),
                optimization_strategy=request.optimization_strategy,
            )
            return OptimizationResult(
                best=fallback_eval,
                pareto=[ParetoItem(evaluation=fallback_eval, rank=0, crowding=0.0)],
                diagnostics=diagnostics,
                matrix_provider=context.matrix_provider,
                population_memory_orders=[list(range(points_count))],
            )

        self._criteria_service.assign_weighted_scores([s.evaluation for s in states], weights)
        fronts = self._rank_and_crowding(states)
        pareto_states = fronts[0] if fronts else states
        pareto_states = sorted(
            pareto_states,
            key=lambda item: (
                0 if item.evaluation.metrics.feasible else 1,
                item.evaluation.metrics.objective_score,
            ),
        )
        if use_pareto:
            best_state = min(
                pareto_states,
                key=lambda item: (
                    0 if item.evaluation.metrics.feasible else 1,
                    -self._crowding_for_sort(item.crowding),
                    item.evaluation.metrics.objective_score,
                ),
            )
        else:
            best_state = min(
                states,
                key=lambda item: (
                    0 if item.evaluation.metrics.feasible else 1,
                    item.evaluation.metrics.objective_score,
                ),
            )

        final_selection = self._select_final_candidate(
            request=request,
            context=context,
            weights=weights,
            fuel_prices=fuel_prices,
            candidate_states=states,
            original_best_state=best_state,
            middle_indices=middle_indices,
            fix_ends=fix_ends,
            points_count=points_count,
        )
        best_state = final_selection.state

        population_memory_orders = self._population_memory_orders(
            states,
            fix_ends=fix_ends,
            points_count=points_count,
            max_size=min(population_size, max(24, request.max_alternatives * 4)),
        )
        diagnostics = OptimizationDiagnostics(
            mode=request.optimize_mode,
            optimization_active=True,
            optimization_reason=None,
            score_mode=ScoreMode.population_normalized,
            generations=generations,
            population_size=population_size,
            crossover_rate=request.crossover_rate,
            mutation_rate=request.mutation_rate,
            stagnation_generations=stagnation_generations,
            evaluated_solutions=evaluated_solutions,
            pareto_solutions=len(pareto_states),
            warm_start_solutions=len(request.warm_start_orders),
            population_memory_solutions=len(population_memory_orders),
            repaired_solutions=repaired_solutions,
            forbidden_edges=len(forbidden_edges),
            forbidden_edges_count=len(forbidden_edges),
            unrepairable_candidates=unrepairable_candidates,
            infrastructure_violations_count=len(forbidden_edges),
            baseline_guard_applied=final_selection.baseline_guard_applied,
            original_best_score=final_selection.original_best_score,
            baseline_score=final_selection.baseline_score,
            baseline_guard_reason=final_selection.baseline_guard_reason,
            final_selected_from=final_selection.final_selected_from,
            final_selection_reason=final_selection.final_selection_reason,
            optimization_strategy=request.optimization_strategy,
            accepted_tradeoff=(
                request.optimization_strategy.value != "strict"
                and not self._does_not_regress_key_metrics(
                    final_selection.baseline_evaluation.metrics,
                    best_state.evaluation.metrics,
                )
                and self._is_candidate_acceptable_for_strategy(
                    request,
                    final_selection.baseline_evaluation.metrics,
                    best_state.evaluation.metrics,
                )
            ),
            rejected_regression_metrics=self._key_metric_regressions(
                final_selection.baseline_evaluation.metrics,
                best_state.evaluation.metrics,
            ),
        )
        logger.warning(
            "GA optimization finished: total_ms=%.3f evaluation_ms=%.3f evaluated_solutions=%d cache_size=%d parallel_workers=%d parallel_batches=%d parallel_evaluations=%d repaired=%d unrepairable=%d stagnation_generations=%d",
            (perf_counter() - optimizer_start) * 1000.0,
            evaluation_ms,
            evaluated_solutions,
            len(eval_cache),
            worker_count,
            parallel_batches,
            parallel_evaluations,
            repaired_solutions,
            unrepairable_candidates,
            stagnation_generations,
        )
        pareto_items = [
            ParetoItem(evaluation=state.evaluation, rank=state.rank, crowding=state.crowding)
            for state in pareto_states[: request.max_alternatives]
        ]
        return OptimizationResult(
            best=best_state.evaluation,
            pareto=pareto_items,
            diagnostics=diagnostics,
            matrix_provider=context.matrix_provider,
            population_memory_orders=population_memory_orders,
        )

    def _single_candidate_result(
        self,
        request: RouteRequest,
        context: OptimizationContext,
        evaluation: CandidateEvaluation,
        optimization_active: bool,
        optimization_reason: OptimizationReason | None,
        score_mode: ScoreMode,
        evaluated_solutions: int = 1,
        forbidden_edges: int = 0,
    ) -> OptimizationResult:
        population_memory_orders = [evaluation.order_indices]
        diagnostics = OptimizationDiagnostics(
            mode=request.optimize_mode,
            optimization_active=optimization_active,
            optimization_reason=optimization_reason,
            score_mode=score_mode,
            generations=1,
            population_size=1,
            crossover_rate=request.crossover_rate,
            mutation_rate=request.mutation_rate,
            stagnation_generations=0,
            evaluated_solutions=evaluated_solutions,
            pareto_solutions=1,
            warm_start_solutions=len(request.warm_start_orders),
            population_memory_solutions=len(population_memory_orders),
            forbidden_edges=forbidden_edges,
            forbidden_edges_count=forbidden_edges,
            infrastructure_violations_count=forbidden_edges,
            original_best_score=evaluation.metrics.objective_score,
            baseline_score=evaluation.metrics.objective_score,
            final_selected_from="baseline"
            if evaluation.order_indices == list(range(len(context.points)))
            else "optimizer",
            final_selection_reason="single_candidate",
            optimization_strategy=request.optimization_strategy,
        )
        return OptimizationResult(
            best=evaluation,
            pareto=[ParetoItem(evaluation=evaluation, rank=0, crowding=0.0)],
            diagnostics=diagnostics,
            matrix_provider=context.matrix_provider,
            population_memory_orders=population_memory_orders,
        )

    def _initial_population(
        self,
        middle_indices: list[int],
        population_size: int,
        rng: random.Random,
        context: OptimizationContext,
        fix_ends: bool,
        points_count: int,
        warm_start_orders: list[list[int]] | None = None,
        forbidden_edges: set[tuple[int, int]] | None = None,
    ) -> list[tuple[int, ...]]:
        seed_genomes: list[tuple[int, ...]] = [tuple(middle_indices)]
        for order in warm_start_orders or []:
            genome = self._genome_from_order(order, middle_indices, fix_ends, points_count)
            if genome is not None:
                seed_genomes.append(genome)
        nearest = self._nearest_neighbor_seed(middle_indices, context, fix_ends, points_count, forbidden_edges)
        if nearest:
            seed_genomes.append(tuple(nearest))
        if forbidden_edges:
            seed_genomes = [
                self._repair_genome(genome, fix_ends, points_count, forbidden_edges)
                for genome in seed_genomes
            ]

        seen = set()
        deduped_seed_genomes: list[tuple[int, ...]] = []
        for genome in seed_genomes:
            if genome in seen:
                continue
            seen.add(genome)
            deduped_seed_genomes.append(genome)
        seed_genomes = deduped_seed_genomes
        attempts = 0
        while len(seed_genomes) < population_size:
            candidate = list(middle_indices)
            rng.shuffle(candidate)
            tup = self._repair_genome(tuple(candidate), fix_ends, points_count, forbidden_edges or set())
            attempts += 1
            if tup in seen and attempts < population_size * 6:
                continue
            if tup not in seen:
                seen.add(tup)
            seed_genomes.append(tup)
        return seed_genomes

    @staticmethod
    def _genome_from_order(
        order: list[int],
        middle_indices: list[int],
        fix_ends: bool,
        points_count: int,
    ) -> tuple[int, ...] | None:
        if len(order) != points_count:
            return None
        if sorted(order) != list(range(points_count)):
            return None
        if fix_ends:
            if not order or order[0] != 0 or order[-1] != points_count - 1:
                return None
            genome = tuple(order[1:-1])
        else:
            genome = tuple(order)
        return genome if sorted(genome) == sorted(middle_indices) else None

    @staticmethod
    def _nearest_neighbor_seed(
        middle_indices: list[int],
        context: OptimizationContext,
        fix_ends: bool,
        points_count: int,
        forbidden_edges: set[tuple[int, int]] | None = None,
    ) -> list[int]:
        if not middle_indices:
            return []
        remaining = list(middle_indices)
        current = 0 if fix_ends else remaining[0]
        if not fix_ends:
            remaining = remaining[1:]
        route: list[int] = []
        while remaining:
            next_point = min(
                remaining,
                key=lambda idx: (
                    (current, idx) in (forbidden_edges or set()),
                    context.distance_matrix_km[current][idx]
                    if current < len(context.distance_matrix_km)
                    and idx < len(context.distance_matrix_km[current])
                    else float("inf")
                ),
            )
            remaining.remove(next_point)
            route.append(next_point)
            current = next_point
        if not fix_ends and points_count > 0:
            route = [middle_indices[0]] + route
        return route

    @staticmethod
    def _decode(genome: tuple[int, ...], fix_ends: bool, points_count: int) -> list[int]:
        return _decode_genome(genome, fix_ends, points_count)

    def _parallel_worker_count(self, population_size: int) -> int:
        configured = self._parallel_workers
        if configured is None:
            cpu_count = os.cpu_count() or 1
            configured = max(1, min(cpu_count - 1, 8))
        if configured <= 1 or population_size < self._parallel_min_batch_size:
            return 1
        return max(1, int(configured))

    @staticmethod
    def _parallel_chunksize(tasks_count: int, worker_count: int) -> int:
        if worker_count <= 1:
            return 1
        return max(1, tasks_count // (worker_count * 4))

    def _rank_and_crowding(self, states: list[CandidateState]) -> list[list[CandidateState]]:
        if not states:
            return []
        domination_sets: list[set[int]] = [set() for _ in states]
        dominated_count = [0 for _ in states]
        fronts_idx: list[list[int]] = [[]]

        vectors = [self._criteria_service.dominance_vector(s.evaluation.metrics) for s in states]
        for i in range(len(states)):
            for j in range(len(states)):
                if i == j:
                    continue
                if self._dominates(vectors[i], vectors[j]):
                    domination_sets[i].add(j)
                elif self._dominates(vectors[j], vectors[i]):
                    dominated_count[i] += 1
            if dominated_count[i] == 0:
                states[i].rank = 0
                fronts_idx[0].append(i)

        front = 0
        while front < len(fronts_idx) and fronts_idx[front]:
            next_front: list[int] = []
            for p in fronts_idx[front]:
                for q in domination_sets[p]:
                    dominated_count[q] -= 1
                    if dominated_count[q] == 0:
                        states[q].rank = front + 1
                        next_front.append(q)
            front += 1
            if next_front:
                fronts_idx.append(next_front)

        fronts: list[list[CandidateState]] = [[states[i] for i in front_indices] for front_indices in fronts_idx if front_indices]
        for front_states in fronts:
            self._compute_crowding(front_states)
        return fronts

    @staticmethod
    def _compute_crowding(front: list[CandidateState]) -> None:
        if not front:
            return
        for candidate in front:
            candidate.crowding = 0.0
        if len(front) <= 2:
            for candidate in front:
                candidate.crowding = float("inf")
            return

        objectives = [
            lambda item: item.evaluation.metrics.distance_km,
            lambda item: item.evaluation.metrics.duration_min,
            lambda item: item.evaluation.metrics.operational_cost,
            lambda item: item.evaluation.metrics.constraint_penalty,
        ]
        for objective in objectives:
            sorted_front = sorted(front, key=objective)
            sorted_front[0].crowding = float("inf")
            sorted_front[-1].crowding = float("inf")
            min_value = objective(sorted_front[0])
            max_value = objective(sorted_front[-1])
            if max_value - min_value <= 1e-12:
                continue
            for idx in range(1, len(sorted_front) - 1):
                if sorted_front[idx].crowding == float("inf"):
                    continue
                next_value = objective(sorted_front[idx + 1])
                prev_value = objective(sorted_front[idx - 1])
                sorted_front[idx].crowding += (next_value - prev_value) / (max_value - min_value)

    @staticmethod
    def _dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
        no_worse = True
        strictly_better = False
        for av, bv in zip(a, b):
            if av > bv:
                no_worse = False
                break
            if av < bv:
                strictly_better = True
        return no_worse and strictly_better

    @staticmethod
    def _select_next_generation(fronts: list[list[CandidateState]], population_size: int) -> list[CandidateState]:
        selected: list[CandidateState] = []
        for front in fronts:
            if len(selected) + len(front) <= population_size:
                selected.extend(front)
            else:
                remaining = population_size - len(selected)
                selected.extend(sorted(front, key=lambda item: item.crowding, reverse=True)[:remaining])
                break
        return selected

    @staticmethod
    def _select_next_generation_weighted(states: list[CandidateState], population_size: int) -> list[CandidateState]:
        return sorted(
            states,
            key=lambda item: (
                0 if item.evaluation.metrics.feasible else 1,
                item.evaluation.metrics.objective_score,
            ),
        )[:population_size]

    def _preserve_elites(
        self,
        *,
        selected: list[CandidateState],
        elite_genomes: set[tuple[int, ...]],
        evaluate_genome: Callable[[tuple[int, ...]], CandidateEvaluation],
        weights: CriteriaWeights,
        population_size: int,
        use_pareto: bool,
    ) -> list[CandidateState]:
        if not selected or not elite_genomes:
            return selected
        present = {state.genome for state in selected}
        result = list(selected)
        for genome in elite_genomes:
            if genome in present:
                continue
            elite_state = CandidateState(genome=genome, evaluation=evaluate_genome(genome))
            if len(result) < population_size:
                result.append(elite_state)
            else:
                worst_idx = self._worst_state_index(result)
                result[worst_idx] = elite_state
            present.add(genome)
        self._criteria_service.assign_weighted_scores([state.evaluation for state in result], weights)
        if use_pareto:
            self._rank_and_crowding(result)
        return result

    @staticmethod
    def _worst_state_index(states: list[CandidateState]) -> int:
        return max(
            range(len(states)),
            key=lambda idx: (
                1 if not states[idx].evaluation.metrics.feasible else 0,
                states[idx].evaluation.metrics.objective_score,
                states[idx].evaluation.metrics.constraint_penalty,
            ),
        )

    def _select_final_candidate(
        self,
        *,
        request: RouteRequest,
        context: OptimizationContext,
        weights: CriteriaWeights,
        fuel_prices: FuelPriceSnapshot,
        candidate_states: list[CandidateState],
        original_best_state: CandidateState,
        middle_indices: list[int],
        fix_ends: bool,
        points_count: int,
    ) -> FinalSelection:
        baseline_order = list(range(points_count))
        baseline_genome = self._genome_from_order(baseline_order, middle_indices, fix_ends, points_count)
        baseline_evaluation = self._criteria_service.evaluate(baseline_order, request, context, fuel_prices)
        baseline_state = CandidateState(
            genome=baseline_genome or tuple(baseline_order),
            evaluation=baseline_evaluation,
        )

        unique_states: list[CandidateState] = []
        seen_orders: set[tuple[int, ...]] = set()
        for state in [original_best_state, *candidate_states, baseline_state]:
            order_key = tuple(state.evaluation.order_indices)
            if order_key in seen_orders:
                continue
            seen_orders.add(order_key)
            unique_states.append(state)

        original_order = tuple(original_best_state.evaluation.order_indices)
        baseline_order_key = tuple(baseline_order)
        scored: list[tuple[CandidateState, float, float, float]] = []
        original_best_pair_score: float | None = None
        original_best_baseline_score: float | None = None
        for state in unique_states:
            if tuple(state.evaluation.order_indices) == baseline_order_key:
                baseline_evaluation.metrics.objective_score = 0.0
                state.evaluation.metrics.objective_score = 0.0
                baseline_score = 0.0
                candidate_score = 0.0
            else:
                self._criteria_service.assign_weighted_scores([baseline_evaluation, state.evaluation], weights)
                baseline_score = baseline_evaluation.metrics.objective_score
                candidate_score = state.evaluation.metrics.objective_score
            score_delta = candidate_score - baseline_score
            scored.append((state, baseline_score, candidate_score, score_delta))
            if tuple(state.evaluation.order_indices) == original_order:
                original_best_pair_score = candidate_score
                original_best_baseline_score = baseline_score

        acceptable = [
            item
            for item in scored
            if item[3] <= 1e-12
            and self._is_candidate_acceptable_for_strategy(
                request,
                baseline_evaluation.metrics,
                item[0].evaluation.metrics,
            )
        ]
        if not acceptable:
            acceptable = [item for item in scored if tuple(item[0].evaluation.order_indices) == baseline_order_key]
        selected_state, _selected_baseline_score, _selected_score, _selected_delta = min(
            acceptable,
            key=lambda item: (
                0 if item[0].evaluation.metrics.feasible else 1,
                item[3],
                item[2],
                item[0].evaluation.metrics.constraint_penalty,
                item[0].evaluation.metrics.duration_min,
                item[0].evaluation.metrics.distance_km,
            ),
        )

        if tuple(selected_state.evaluation.order_indices) == baseline_order_key:
            baseline_evaluation.metrics.objective_score = 0.0
            selected_state.evaluation.metrics.objective_score = 0.0
        else:
            self._criteria_service.assign_weighted_scores([baseline_evaluation, selected_state.evaluation], weights)

        selected_order = tuple(selected_state.evaluation.order_indices)
        final_selected_from = (
            "baseline"
            if selected_order == baseline_order_key
            else "optimizer"
            if selected_order == original_order
            else "alternative"
        )
        original_score = (
            original_best_pair_score
            if original_best_pair_score is not None
            else original_best_state.evaluation.metrics.objective_score
        )
        original_baseline_score = original_best_baseline_score if original_best_baseline_score is not None else 0.0
        baseline_guard_applied = (
            selected_order != original_order
            or original_score - original_baseline_score > 1e-12
            or not self._is_candidate_acceptable_for_strategy(
                request,
                baseline_evaluation.metrics,
                original_best_state.evaluation.metrics,
            )
        )
        original_best_regressions = self._key_metric_regressions(
            baseline_evaluation.metrics,
            original_best_state.evaluation.metrics,
        )
        if final_selected_from == "baseline" and original_order != baseline_order_key:
            reason = (
                "optimizer_best_regressed_key_metrics"
                if original_best_regressions
                else "optimizer_best_worse_than_baseline"
            )
        elif final_selected_from == "alternative":
            reason = "alternative_without_metric_regression"
        elif final_selected_from == "baseline":
            reason = "baseline_selected"
        else:
            reason = "optimizer_best_selected"

        return FinalSelection(
            state=selected_state,
            baseline_evaluation=baseline_evaluation,
            baseline_guard_applied=baseline_guard_applied,
            original_best_score=float(original_score),
            baseline_score=float(baseline_evaluation.metrics.objective_score),
            baseline_guard_reason=reason if baseline_guard_applied else None,
            final_selected_from=final_selected_from,
            final_selection_reason=reason,
        )

    @classmethod
    def _does_not_regress_key_metrics(cls, baseline: RouteMetrics, candidate: RouteMetrics) -> bool:
        return not cls._key_metric_regressions(baseline, candidate)

    @classmethod
    def _is_candidate_acceptable_for_strategy(
        cls,
        request: RouteRequest,
        baseline: RouteMetrics,
        candidate: RouteMetrics,
    ) -> bool:
        if request.optimization_strategy.value == "strict":
            return cls._does_not_regress_key_metrics(baseline, candidate)
        regressions = cls._key_metric_regressions(baseline, candidate)
        if not regressions:
            return True

        tolerance = request.tradeoff_tolerance
        if request.optimization_strategy.value == "balanced":
            duration_gain = cls._improvement_pct(baseline.duration_min, candidate.duration_min)
            distance_gain = cls._improvement_pct(baseline.distance_km, candidate.distance_km)
            major_efficiency_gain = duration_gain >= 15.0 or distance_gain >= 15.0
            max_distance = 25.0 if major_efficiency_gain else 10.0
            max_duration = 15.0 if major_efficiency_gain else 5.0
            max_operational_cost = 20.0 if major_efficiency_gain else 8.0
            allow_penalty = False
        else:
            max_distance = tolerance.max_distance_regression_pct
            max_duration = tolerance.max_duration_regression_pct
            max_operational_cost = tolerance.max_operational_cost_regression_pct
            allow_penalty = tolerance.allow_constraint_penalty_regression

        allowed_regression_pct = {
            "distance_km": max_distance,
            "duration_min": max_duration,
            "operational_cost": max_operational_cost,
        }
        for metric in regressions:
            if metric == "constraint_penalty":
                if not allow_penalty:
                    return False
                continue
            limit = allowed_regression_pct.get(metric, 0.0)
            baseline_value = float(getattr(baseline, metric))
            candidate_value = float(getattr(candidate, metric))
            if cls._regression_pct(baseline_value, candidate_value) > limit + 1e-9:
                return False
        return True

    @classmethod
    def _key_metric_regressions(cls, baseline: RouteMetrics, candidate: RouteMetrics) -> list[str]:
        checks = (
            ("distance_km", baseline.distance_km, candidate.distance_km),
            ("duration_min", baseline.duration_min, candidate.duration_min),
            ("operational_cost", baseline.operational_cost, candidate.operational_cost),
            ("constraint_penalty", baseline.constraint_penalty, candidate.constraint_penalty),
        )
        regressions: list[str] = []
        for key, baseline_value, candidate_value in checks:
            tolerance = max(cls._NON_REGRESSION_EPSILON, abs(float(baseline_value)) * cls._NON_REGRESSION_EPSILON)
            if float(candidate_value) > float(baseline_value) + tolerance:
                regressions.append(key)
        return regressions

    @staticmethod
    def _regression_pct(baseline_value: float, candidate_value: float) -> float:
        if abs(baseline_value) <= 1e-12:
            return 0.0 if candidate_value <= baseline_value else 100.0
        return max(0.0, ((candidate_value - baseline_value) / abs(baseline_value)) * 100.0)

    @staticmethod
    def _improvement_pct(baseline_value: float, candidate_value: float) -> float:
        if abs(baseline_value) <= 1e-12:
            return 0.0
        return max(0.0, ((baseline_value - candidate_value) / abs(baseline_value)) * 100.0)

    @staticmethod
    def _tournament_select(states: list[CandidateState], rng: random.Random) -> CandidateState:
        a = rng.choice(states)
        b = rng.choice(states)
        if a.rank < b.rank:
            return a
        if b.rank < a.rank:
            return b
        if a.crowding > b.crowding:
            return a
        if b.crowding > a.crowding:
            return b
        return a if a.evaluation.metrics.objective_score <= b.evaluation.metrics.objective_score else b

    @staticmethod
    def _tournament_select_weighted(states: list[CandidateState], rng: random.Random) -> CandidateState:
        a = rng.choice(states)
        b = rng.choice(states)
        if a.evaluation.metrics.feasible and not b.evaluation.metrics.feasible:
            return a
        if b.evaluation.metrics.feasible and not a.evaluation.metrics.feasible:
            return b
        return a if a.evaluation.metrics.objective_score <= b.evaluation.metrics.objective_score else b

    @staticmethod
    def _crowding_for_sort(crowding: float) -> float:
        if crowding == float("inf"):
            return 1e12
        return crowding

    @staticmethod
    def _pmx_crossover(
        parent_a: tuple[int, ...],
        parent_b: tuple[int, ...],
        rng: random.Random,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if len(parent_a) != len(parent_b) or len(parent_a) < 2:
            return parent_a, parent_b
        left = rng.randint(0, len(parent_a) - 2)
        right = rng.randint(left + 1, len(parent_a) - 1)

        def build_child(first: tuple[int, ...], second: tuple[int, ...]) -> tuple[int, ...]:
            child: list[int | None] = [None] * len(first)
            child[left : right + 1] = first[left : right + 1]
            second_positions = {gene: idx for idx, gene in enumerate(second)}

            for idx in range(left, right + 1):
                mapped_gene = second[idx]
                if mapped_gene in child:
                    continue

                target_idx = idx
                while child[target_idx] is not None:
                    target_idx = second_positions[first[target_idx]]
                child[target_idx] = mapped_gene

            for idx, gene in enumerate(second):
                if child[idx] is None:
                    child[idx] = gene

            return tuple(int(gene) for gene in child if gene is not None)

        return build_child(parent_a, parent_b), build_child(parent_b, parent_a)

    @staticmethod
    def _mutate(genome: tuple[int, ...], rng: random.Random, mutation_rate: float) -> tuple[int, ...]:
        if len(genome) < 2 or rng.random() > mutation_rate:
            return genome
        genes = list(genome)
        mutation_kind = rng.random()
        if mutation_kind < (1.0 / 3.0):
            i, j = rng.sample(range(len(genes)), 2)
            genes[i], genes[j] = genes[j], genes[i]
        elif mutation_kind < (2.0 / 3.0):
            genes = list(RouteOptimizer._insertion_mutation(tuple(genes), rng))
        else:
            i, j = sorted(rng.sample(range(len(genes)), 2))
            genes[i : j + 1] = reversed(genes[i : j + 1])
        return tuple(genes)

    @staticmethod
    def _insertion_mutation(genome: tuple[int, ...], rng: random.Random) -> tuple[int, ...]:
        if len(genome) < 2:
            return genome
        genes = list(genome)
        source_idx, target_idx = rng.sample(range(len(genes)), 2)
        gene = genes.pop(source_idx)
        genes.insert(target_idx, gene)
        return tuple(genes)

    @staticmethod
    def _population_memory_orders(
        states: list[CandidateState],
        *,
        fix_ends: bool,
        points_count: int,
        max_size: int,
    ) -> list[list[int]]:
        ordered_states = sorted(
            states,
            key=lambda item: (
                0 if item.evaluation.metrics.feasible else 1,
                item.rank,
                item.evaluation.metrics.objective_score,
                -RouteOptimizer._crowding_for_sort(item.crowding),
            ),
        )
        memory: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()
        for state in ordered_states:
            order = RouteOptimizer._decode(state.genome, fix_ends, points_count)
            key = tuple(order)
            if key in seen:
                continue
            seen.add(key)
            memory.append(order)
            if len(memory) >= max_size:
                break
        return memory

    @staticmethod
    def _repair_genome(
        genome: tuple[int, ...],
        fix_ends: bool,
        points_count: int,
        forbidden_edges: set[tuple[int, int]] | None,
    ) -> tuple[int, ...]:
        if not forbidden_edges or len(genome) < 2:
            return genome
        best = genome
        best_score = RouteOptimizer._bad_edge_count(
            RouteOptimizer._decode(best, fix_ends, points_count),
            forbidden_edges,
        )
        if best_score == 0:
            return best

        max_passes = min(6, len(genome))
        for _ in range(max_passes):
            candidate_best = best
            candidate_score = best_score
            for i in range(len(best)):
                for j in range(len(best)):
                    if i == j:
                        continue
                    swapped = RouteOptimizer._swap_positions(best, i, j)
                    swapped_score = RouteOptimizer._bad_edge_count(
                        RouteOptimizer._decode(swapped, fix_ends, points_count),
                        forbidden_edges,
                    )
                    if swapped_score < candidate_score:
                        candidate_best = swapped
                        candidate_score = swapped_score
                    inserted = RouteOptimizer._move_position(best, i, j)
                    inserted_score = RouteOptimizer._bad_edge_count(
                        RouteOptimizer._decode(inserted, fix_ends, points_count),
                        forbidden_edges,
                    )
                    if inserted_score < candidate_score:
                        candidate_best = inserted
                        candidate_score = inserted_score
                if candidate_score == 0:
                    break
            if candidate_score >= best_score:
                break
            best = candidate_best
            best_score = candidate_score
            if best_score == 0:
                break
        return best

    @staticmethod
    def _swap_positions(genome: tuple[int, ...], i: int, j: int) -> tuple[int, ...]:
        genes = list(genome)
        genes[i], genes[j] = genes[j], genes[i]
        return tuple(genes)

    @staticmethod
    def _move_position(genome: tuple[int, ...], source_idx: int, target_idx: int) -> tuple[int, ...]:
        genes = list(genome)
        gene = genes.pop(source_idx)
        genes.insert(target_idx, gene)
        return tuple(genes)

    @staticmethod
    def _bad_edge_count(order: list[int], forbidden_edges: set[tuple[int, int]]) -> int:
        return sum(1 for i, j in zip(order, order[1:], strict=False) if (i, j) in forbidden_edges)

    @staticmethod
    def _forbidden_edges(request: RouteRequest, context: OptimizationContext) -> set[tuple[int, int]]:
        forbidden: set[tuple[int, int]] = set()
        dimensions = request.vehicle_dimensions
        size = len(context.points)
        for i in range(size):
            for j in range(size):
                if i == j:
                    continue
                if not RouteOptimizer._safe_bool_matrix_value(context.infrastructure_access_matrix, i, j):
                    forbidden.add((i, j))
                    continue
                if not RouteOptimizer._safe_bool_matrix_value(context.temporal_access_matrix, i, j):
                    forbidden.add((i, j))
                    continue
                if RouteOptimizer._limit_exceeded(dimensions.height_m, context.height_clearance_matrix_m, i, j):
                    forbidden.add((i, j))
                    continue
                if RouteOptimizer._limit_exceeded(dimensions.weight_t, context.weight_limit_matrix_t, i, j):
                    forbidden.add((i, j))
                    continue
                if RouteOptimizer._limit_exceeded(dimensions.width_m, context.width_limit_matrix_m, i, j):
                    forbidden.add((i, j))
                    continue
                if RouteOptimizer._limit_exceeded(dimensions.length_m, context.length_limit_matrix_m, i, j):
                    forbidden.add((i, j))
        return forbidden

    @staticmethod
    def _limit_exceeded(
        required_value: float | None,
        matrix: list[list[float | None]],
        i: int,
        j: int,
    ) -> bool:
        if required_value is None or i >= len(matrix) or j >= len(matrix[i]) or matrix[i][j] is None:
            return False
        return required_value > float(matrix[i][j])

    @staticmethod
    def _safe_bool_matrix_value(matrix: list[list[bool]], i: int, j: int) -> bool:
        if i < len(matrix) and j < len(matrix[i]):
            return bool(matrix[i][j])
        return True
