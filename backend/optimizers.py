from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from models import (
    EvaluationBreakdown,
    ExperimentConfig,
    IterationLog,
    OptimizerType,
    ParticleSnapshot,
)

# Тип функции оценки точки X_i (например, метод EvaluationFacade.evaluate)
EvaluateFn = Callable[[List[float]], EvaluationBreakdown]


def _round_vec(vec: List[float], digits: int = 4) -> List[float]:
    """Возвращает округленную копию вектора для безопасного логирования в JSON."""
    return [round(float(v), digits) for v in vec]


def _clamp(value: float, low: float, high: float) -> float:
    """Ограничивает числовое значение отрезком [low, high]."""
    return max(low, min(high, value))


@dataclass
class _AgentState:
    """Внутреннее состояние агента (частицы или особи) в пространстве поиска."""

    agent_id: int
    position: List[float]
    velocity: List[float]
    best_position: List[float]
    best_cost: float
    best_fitness: float
    current_evaluation: EvaluationBreakdown

    def to_snapshot(self) -> ParticleSnapshot:
        """Создает неизменяемый снимок состояния агента для истории и визуализации."""
        return ParticleSnapshot(
            particle_id=self.agent_id,
            position=_round_vec(self.position),
            velocity=_round_vec(self.velocity),
            best_position=_round_vec(self.best_position),
            best_fitness=round(self.best_fitness, 4),
            current_evaluation=self.current_evaluation,
        )


class BaseOptimizer(ABC):
    """Абстрактный базовый класс алгоритмов оптимизации (паттерн Strategy)."""

    optimizer_type: OptimizerType

    def __init__(
        self,
        num_particles: int,
        bounds: List[Tuple[float, float]],
        evaluate_fn: EvaluateFn,
        seed: Optional[int] = 42,
    ) -> None:
        if num_particles < 2:
            raise ValueError("num_particles должно быть не менее 2.")
        if not bounds:
            raise ValueError("bounds не может быть пустым.")
        for idx, (low, high) in enumerate(bounds):
            if low > high:
                raise ValueError(
                    f"Некорректные границы по координате {idx}: ({low}, {high})."
                )

        self.num_particles = num_particles
        self.bounds = list(bounds)
        self.dim = len(self.bounds)
        self.evaluate_fn = evaluate_fn
        self.seed = seed

        self._rng = random.Random(seed)
        self._agents: List[_AgentState] = []
        self._initialized: bool = False
        self.iteration: int = 0

        self.global_best_position: List[float] = []
        self.global_best_cost: float = float("inf")
        self.global_best_fitness: float = 0.0
        self.global_best_evaluation: Optional[EvaluationBreakdown] = None
        self.history: List[IterationLog] = []

    def _random_position(self) -> List[float]:
        """Генерирует случайный вектор координат в пределах bounds."""
        return [self._rng.uniform(low, high) for low, high in self.bounds]

    def _zero_velocity(self) -> List[float]:
        """Нулевой вектор скорости (для алгоритмов без инерции)."""
        return [0.0] * self.dim

    def _initial_velocity(self) -> List[float]:
        """Начальный вектор скорости агента (переопределяется в PSO)."""
        return self._zero_velocity()

    def _update_global_best(
        self, position: List[float], evaluation: EvaluationBreakdown
    ) -> None:
        """Монотонно обновляет глобальный рекорд G_best."""
        if (
            self.global_best_evaluation is None
            or evaluation.cost < self.global_best_cost
        ):
            self.global_best_cost = evaluation.cost
            self.global_best_fitness = evaluation.fitness
            self.global_best_position = list(position)
            self.global_best_evaluation = evaluation

    def _record_iteration(self, iteration_time_sec: float) -> IterationLog:
        """Формирует и сохраняет лог текущей итерации."""
        assert self.global_best_evaluation is not None
        snapshots = [agent.to_snapshot() for agent in self._agents]
        mean_fitness = round(
            sum(s.current_evaluation.fitness for s in snapshots) / len(snapshots),
            4,
        )
        log = IterationLog(
            iteration=self.iteration,
            global_best_fitness=round(self.global_best_fitness, 4),
            global_best_cost=round(self.global_best_cost, 4),
            global_best_position=_round_vec(self.global_best_position),
            global_best_evaluation=self.global_best_evaluation,
            mean_fitness=mean_fitness,
            iteration_time_sec=round(iteration_time_sec, 3),
            particles=snapshots,
        )
        self.history.append(log)
        return log

    def initialize(self) -> IterationLog:
        """Инициализирует популяцию (итерация 0) и вычисляет начальные оценки."""
        self._rng = random.Random(self.seed)
        self._agents = []
        self.history = []
        self.iteration = 0
        self.global_best_position = []
        self.global_best_cost = float("inf")
        self.global_best_fitness = 0.0
        self.global_best_evaluation = None

        iter_time = 0.0
        for agent_id in range(self.num_particles):
            pos = self._random_position()
            vel = self._initial_velocity()
            ev = self.evaluate_fn(pos)
            iter_time += ev.eval_time_sec

            agent = _AgentState(
                agent_id=agent_id,
                position=pos,
                velocity=vel,
                best_position=list(pos),
                best_cost=ev.cost,
                best_fitness=ev.fitness,
                current_evaluation=ev,
            )
            self._agents.append(agent)
            self._update_global_best(pos, ev)

        self._initialized = True
        return self._record_iteration(iter_time)

    @abstractmethod
    def _advance_agents(self) -> float:
        """Выполняет один шаг алгоритма и возвращает суммарное время оценок на шаге."""

    def step(self) -> IterationLog:
        """Выполняет следующую итерацию оптимизации (при необходимости вызывает initialize)."""
        if not self._initialized:
            self.initialize()

        self.iteration += 1
        iter_time = self._advance_agents()
        return self._record_iteration(iter_time)


class RandomSearchOptimizer(BaseOptimizer):
    """
    Алгоритм 1 (Бейзлайн): случайный поиск (Random Search).
    На каждой итерации генерирует независимые равномерно распределенные точки без памяти.
    """

    optimizer_type = OptimizerType.RANDOM

    def _advance_agents(self) -> float:
        iter_time = 0.0
        for agent in self._agents:
            new_pos = self._random_position()
            ev = self.evaluate_fn(new_pos)
            iter_time += ev.eval_time_sec

            agent.position = new_pos
            agent.velocity = self._zero_velocity()
            agent.current_evaluation = ev

            if ev.cost < agent.best_cost:
                agent.best_cost = ev.cost
                agent.best_fitness = ev.fitness
                agent.best_position = list(new_pos)

            self._update_global_best(new_pos, ev)

        return iter_time


class GeneticAlgorithmOptimizer(BaseOptimizer):
    """
    Алгоритм 2: генетический алгоритм (Genetic Algorithm, GA).
    Реализует элитизм, турнирную селекцию, одноточечный кроссовер и гауссову мутацию.
    """

    optimizer_type = OptimizerType.GA

    def __init__(
        self,
        num_particles: int,
        bounds: List[Tuple[float, float]],
        evaluate_fn: EvaluateFn,
        elitism: int = 2,
        crossover_rate: float = 0.80,
        mutation_rate: float = 0.15,
        mutation_sigma_ratio: float = 0.12,
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__(
            num_particles=num_particles,
            bounds=bounds,
            evaluate_fn=evaluate_fn,
            seed=seed,
        )
        if not (0 <= elitism < num_particles):
            raise ValueError("elitism должно удовлетворять 0 <= elitism < num_particles.")
        if not (0.0 <= crossover_rate <= 1.0):
            raise ValueError("crossover_rate должно лежать в [0.0, 1.0].")
        if not (0.0 <= mutation_rate <= 1.0):
            raise ValueError("mutation_rate должно лежать в [0.0, 1.0].")

        self.elitism = elitism
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_sigma_ratio = mutation_sigma_ratio

    def _tournament_select(self, population: List[_AgentState]) -> _AgentState:
        """Турнирная селекция (k = min(3, len(population)))."""
        k = min(3, len(population))
        contenders = self._rng.sample(population, k)
        return min(contenders, key=lambda agent: agent.current_evaluation.cost)

    def _crossover(
        self, parent_a: List[float], parent_b: List[float]
    ) -> Tuple[List[float], List[float]]:
        """Одноточечный кроссовер двух родительских хромосом."""
        if self.dim <= 1 or self._rng.random() >= self.crossover_rate:
            return list(parent_a), list(parent_b)

        cut = self._rng.randint(1, self.dim - 1)
        child_1 = parent_a[:cut] + parent_b[cut:]
        child_2 = parent_b[:cut] + parent_a[cut:]
        return child_1, child_2

    def _mutate(self, chromosome: List[float]) -> List[float]:
        """Гауссова мутация каждого гена с вероятностью mutation_rate."""
        mutated = list(chromosome)
        for i, (low, high) in enumerate(self.bounds):
            if self._rng.random() < self.mutation_rate:
                sigma = (high - low) * self.mutation_sigma_ratio
                mutated[i] = _clamp(mutated[i] + self._rng.gauss(0.0, sigma), low, high)
        return mutated

    def _advance_agents(self) -> float:
        sorted_pop = sorted(
            self._agents, key=lambda a: a.current_evaluation.cost
        )
        next_agents: List[_AgentState] = []
        iter_time = 0.0

        # 1. Элитизм: переносим лучших особей без искажения и лишнего вызова GPU
        for new_id in range(self.elitism):
            elite = sorted_pop[new_id]
            next_agents.append(
                _AgentState(
                    agent_id=new_id,
                    position=list(elite.position),
                    velocity=self._zero_velocity(),
                    best_position=list(elite.best_position),
                    best_cost=elite.best_cost,
                    best_fitness=elite.best_fitness,
                    current_evaluation=elite.current_evaluation,
                )
            )

        # 2. Генерация потомков до заполнения популяции
        next_id = self.elitism
        while len(next_agents) < self.num_particles:
            parent_1 = self._tournament_select(sorted_pop)
            parent_2 = self._tournament_select(sorted_pop)
            raw_children = self._crossover(parent_1.position, parent_2.position)

            for raw_child in raw_children:
                if len(next_agents) >= self.num_particles:
                    break
                child_pos = self._mutate(raw_child)
                ev = self.evaluate_fn(child_pos)
                iter_time += ev.eval_time_sec

                next_agents.append(
                    _AgentState(
                        agent_id=next_id,
                        position=child_pos,
                        velocity=self._zero_velocity(),
                        best_position=list(child_pos),
                        best_cost=ev.cost,
                        best_fitness=ev.fitness,
                        current_evaluation=ev,
                    )
                )
                self._update_global_best(child_pos, ev)
                next_id += 1

        self._agents = next_agents
        return iter_time


class HybridPSOOptimizer(BaseOptimizer):
    """
    Алгоритм 3 (Предложенный): адаптированный метод роя частиц (PSO).
    Использует коэффициенты инерции w, когнитивную c1 и социальную c2 составляющие,
    ограничение максимальной скорости и упругое отражение от границ пространства поиска.
    """

    optimizer_type = OptimizerType.PSO

    def __init__(
        self,
        num_particles: int,
        bounds: List[Tuple[float, float]],
        evaluate_fn: EvaluateFn,
        w: float = 0.72,
        c1: float = 1.49,
        c2: float = 1.49,
        max_vel_ratio: float = 0.25,
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__(
            num_particles=num_particles,
            bounds=bounds,
            evaluate_fn=evaluate_fn,
            seed=seed,
        )
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_vel_ratio = max_vel_ratio

    def _initial_velocity(self) -> List[float]:
        return [
            self._rng.uniform(-(high - low) * 0.1, (high - low) * 0.1)
            for low, high in self.bounds
        ]

    def _advance_agents(self) -> float:
        iter_time = 0.0
        for agent in self._agents:
            for i, (low, high) in enumerate(self.bounds):
                r1 = self._rng.random()
                r2 = self._rng.random()

                cognitive = self.c1 * r1 * (agent.best_position[i] - agent.position[i])
                social = self.c2 * r2 * (self.global_best_position[i] - agent.position[i])

                new_vel = self.w * agent.velocity[i] + cognitive + social
                max_vel = (high - low) * self.max_vel_ratio
                new_vel = _clamp(new_vel, -max_vel, max_vel)

                new_pos = agent.position[i] + new_vel
                if new_pos < low:
                    new_pos = low
                    new_vel *= -0.5
                elif new_pos > high:
                    new_pos = high
                    new_vel *= -0.5

                agent.velocity[i] = new_vel
                agent.position[i] = new_pos

            ev = self.evaluate_fn(agent.position)
            iter_time += ev.eval_time_sec
            agent.current_evaluation = ev

            if ev.cost < agent.best_cost:
                agent.best_cost = ev.cost
                agent.best_fitness = ev.fitness
                agent.best_position = list(agent.position)

            self._update_global_best(agent.position, ev)

        return iter_time


def create_optimizer(
    config: ExperimentConfig,
    evaluate_fn: EvaluateFn,
    seed: Optional[int] = None,
) -> BaseOptimizer:
    """Фабричный метод создания алгоритма оптимизации из конфигурации эксперимента."""
    effective_seed = config.seed if seed is None else seed
    bounds = config.get_search_bounds()

    if config.optimizer == OptimizerType.RANDOM:
        return RandomSearchOptimizer(
            num_particles=config.num_particles,
            bounds=bounds,
            evaluate_fn=evaluate_fn,
            seed=effective_seed,
        )

    if config.optimizer == OptimizerType.GA:
        return GeneticAlgorithmOptimizer(
            num_particles=config.num_particles,
            bounds=bounds,
            evaluate_fn=evaluate_fn,
            elitism=config.ga_elitism,
            crossover_rate=config.ga_crossover_rate,
            mutation_rate=config.ga_mutation_rate,
            seed=effective_seed,
        )

    if config.optimizer == OptimizerType.PSO:
        return HybridPSOOptimizer(
            num_particles=config.num_particles,
            bounds=bounds,
            evaluate_fn=evaluate_fn,
            w=config.pso_w,
            c1=config.pso_c1,
            c2=config.pso_c2,
            seed=effective_seed,
        )

    raise ValueError(f"Неподдерживаемый тип оптимизатора: {config.optimizer}")
