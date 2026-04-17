from __future__ import annotations

from dataclasses import dataclass
import random

from app.domain.models import (
    CriteriaWeights,
    OptimizationDiagnostics,
    OptimizationMode,
    OptimizationReason,
    RouteRequest,
    ScoreMode,
)
from app.services.context_service import OptimizationContext
from app.services.criteria_service import CandidateEvaluation, CriteriaService
from app.services.fuel_cost import FuelPriceSnapshot


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
class OptimizationResult:
    best: CandidateEvaluation
    pareto: list[ParetoItem]
    diagnostics: OptimizationDiagnostics
    matrix_provider: str


class RouteOptimizer:
    def __init__(self, criteria_service: CriteriaService) -> None:
        self._criteria_service = criteria_service

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
            )

        population_size = max(24, request.population_size)
        generations = max(20, request.generations)
        eval_cache: dict[tuple[int, ...], CandidateEvaluation] = {}
        evaluated_solutions = 0

        def evaluate_genome(genome: tuple[int, ...]) -> CandidateEvaluation:
            nonlocal evaluated_solutions
            cached = eval_cache.get(genome)
            if cached is not None:
                return cached
            order = self._decode(genome, fix_ends, points_count)
            evaluation = self._criteria_service.evaluate(order, request, context, fuel_prices)
            eval_cache[genome] = evaluation
            evaluated_solutions += 1
            return evaluation

        population_genomes = self._initial_population(
            middle_indices=middle_indices,
            population_size=population_size,
            rng=rng,
            context=context,
            fix_ends=fix_ends,
            points_count=points_count,
        )

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
                    child1, child2 = self._order_crossover(p1.genome, p2.genome, rng)
                child1 = self._mutate(child1, rng, request.mutation_rate)
                child2 = self._mutate(child2, rng, request.mutation_rate)
                offspring_genomes.append(child1)
                if len(offspring_genomes) < population_size:
                    offspring_genomes.append(child2)

            offspring_states = [CandidateState(genome=g, evaluation=evaluate_genome(g)) for g in offspring_genomes]
            merged = states + offspring_states
            self._criteria_service.assign_weighted_scores([s.evaluation for s in merged], weights)
            if use_pareto:
                merged_fronts = self._rank_and_crowding(merged)
                states = self._select_next_generation(merged_fronts, population_size)
            else:
                states = self._select_next_generation_weighted(merged, population_size)

            generation_best = min([s.evaluation.metrics.objective_score for s in states]) if states else float("inf")
            if generation_best + 1e-8 < best_score:
                best_score = generation_best
                stagnation_generations = 0
            else:
                stagnation_generations += 1

            if use_pareto:
                self._rank_and_crowding(states)

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
            )
            return OptimizationResult(
                best=fallback_eval,
                pareto=[ParetoItem(evaluation=fallback_eval, rank=0, crowding=0.0)],
                diagnostics=diagnostics,
                matrix_provider=context.matrix_provider,
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
    ) -> OptimizationResult:
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
        )
        return OptimizationResult(
            best=evaluation,
            pareto=[ParetoItem(evaluation=evaluation, rank=0, crowding=0.0)],
            diagnostics=diagnostics,
            matrix_provider=context.matrix_provider,
        )

    def _initial_population(
        self,
        middle_indices: list[int],
        population_size: int,
        rng: random.Random,
        context: OptimizationContext,
        fix_ends: bool,
        points_count: int,
    ) -> list[tuple[int, ...]]:
        seed_genomes: list[tuple[int, ...]] = [tuple(middle_indices)]
        nearest = self._nearest_neighbor_seed(middle_indices, context, fix_ends, points_count)
        if nearest:
            seed_genomes.append(tuple(nearest))

        seen = set(seed_genomes)
        attempts = 0
        while len(seed_genomes) < population_size:
            candidate = list(middle_indices)
            rng.shuffle(candidate)
            tup = tuple(candidate)
            attempts += 1
            if tup in seen and attempts < population_size * 6:
                continue
            if tup not in seen:
                seen.add(tup)
            seed_genomes.append(tup)
        return seed_genomes

    @staticmethod
    def _nearest_neighbor_seed(
        middle_indices: list[int],
        context: OptimizationContext,
        fix_ends: bool,
        points_count: int,
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
        middle = list(genome)
        if fix_ends:
            return [0] + middle + [points_count - 1]
        return middle

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
            lambda item: item.evaluation.metrics.fuel_cost,
            lambda item: item.evaluation.metrics.co2_kg,
            lambda item: item.evaluation.metrics.congestion_index,
            lambda item: item.evaluation.metrics.weather_risk,
            lambda item: (1.0 - item.evaluation.metrics.reliability_score),
            lambda item: item.evaluation.metrics.safety_risk,
            lambda item: item.evaluation.metrics.toll_cost,
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
    def _order_crossover(
        parent_a: tuple[int, ...],
        parent_b: tuple[int, ...],
        rng: random.Random,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if len(parent_a) != len(parent_b) or len(parent_a) < 2:
            return parent_a, parent_b
        left = rng.randint(0, len(parent_a) - 2)
        right = rng.randint(left + 1, len(parent_a) - 1)

        def build_child(first: tuple[int, ...], second: tuple[int, ...]) -> tuple[int, ...]:
            child = [None] * len(first)
            child[left : right + 1] = list(first[left : right + 1])
            used = {item for item in child if item is not None}
            pos = (right + 1) % len(first)
            for gene in second:
                if gene in used:
                    continue
                while child[pos] is not None:
                    pos = (pos + 1) % len(first)
                child[pos] = gene
                used.add(gene)
            return tuple([int(g) for g in child if g is not None])

        return build_child(parent_a, parent_b), build_child(parent_b, parent_a)

    @staticmethod
    def _mutate(genome: tuple[int, ...], rng: random.Random, mutation_rate: float) -> tuple[int, ...]:
        if len(genome) < 2 or rng.random() > mutation_rate:
            return genome
        genes = list(genome)
        if rng.random() < 0.5:
            i, j = rng.sample(range(len(genes)), 2)
            genes[i], genes[j] = genes[j], genes[i]
        else:
            i, j = sorted(rng.sample(range(len(genes)), 2))
            genes[i : j + 1] = reversed(genes[i : j + 1])
        return tuple(genes)
