from __future__ import annotations

import copy
import json
import math
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evaluator import EvaluationFacade
from models import (
    ExperimentConfig,
    ExperimentResult,
    IterationLog,
    MacroMetrics,
    OptimizerType,
    PromptStrategyType,
    SchedulerType,
)
from optimizers import BaseOptimizer, create_optimizer


def _sample_std(values: List[float]) -> float:
    """Выборочное стандартное отклонение (с поправкой Бесселя N - 1)."""
    if len(values) <= 1:
        return 0.0
    mean_val = sum(values) / len(values)
    variance = sum((v - mean_val) ** 2 for v in values) / (len(values) - 1)
    return round(math.sqrt(variance), 4)


def _clone_config(
    base_config: ExperimentConfig,
    **overrides: Any,
) -> ExperimentConfig:
    """Создает независимую копию ExperimentConfig с переопределением выбранных полей."""
    cloned = replace(
        base_config,
        metric_weights=replace(base_config.metric_weights),
        slot_dictionaries=copy.deepcopy(base_config.slot_dictionaries),
        **overrides,
    )
    return cloned


@dataclass(frozen=True)
class ComparisonPreset:
    """Описание одной конфигурации в сравнительном бенчмарке (Таблица 3.4)."""

    preset_id: str
    label: str
    optimizer: OptimizerType
    prompt_strategy: PromptStrategyType
    scheduler: SchedulerType


class BaseExperimentObserver(ABC):
    """Абстрактный наблюдатель за ходом эксперимента (паттерн Observer)."""

    @abstractmethod
    def on_experiment_start(self, experiment_id: str, config: ExperimentConfig) -> None:
        """Вызывается перед запуском эксперимента."""

    @abstractmethod
    def on_iteration(self, experiment_id: str, log: IterationLog) -> None:
        """Вызывается после завершения каждой итерации оптимизации."""

    @abstractmethod
    def on_experiment_end(self, result: ExperimentResult) -> None:
        """Вызывается после завершения эксперимента и подсчета макро-метрик."""


class MetricsLoggerObserver(BaseExperimentObserver):
    """Наблюдатель, собирающий компактную сводку кривой сходимости в памяти."""

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def on_experiment_start(self, experiment_id: str, config: ExperimentConfig) -> None:
        self.records.clear()

    def on_iteration(self, experiment_id: str, log: IterationLog) -> None:
        self.records.append(
            {
                "experiment_id": experiment_id,
                "iteration": log.iteration,
                "global_best_fitness": log.global_best_fitness,
                "mean_fitness": log.mean_fitness,
                "iteration_time_sec": log.iteration_time_sec,
            }
        )

    def on_experiment_end(self, result: ExperimentResult) -> None:
        return None


class ExperimentFactory:
    """Фабрика сборки компонентов эксперимента и стандартных пресетов (паттерн Factory Method)."""

    @staticmethod
    def build(
        config: ExperimentConfig,
        seed: Optional[int] = None,
    ) -> Tuple[EvaluationFacade, BaseOptimizer]:
        """Создает согласованную пару (фасад оценки, алгоритм оптимизации)."""
        effective_seed = config.seed if seed is None else seed
        run_config = (
            _clone_config(config, seed=effective_seed)
            if effective_seed != config.seed
            else config
        )
        facade = EvaluationFacade(run_config)
        optimizer = create_optimizer(
            config=run_config,
            evaluate_fn=facade.evaluate,
            seed=effective_seed,
        )
        return facade, optimizer

    @staticmethod
    def default_comparison_presets() -> List[ComparisonPreset]:
        """Четыре эталонные связки для сравнительного анализа (Глава 3.4)."""
        return [
            ComparisonPreset(
                preset_id="baseline_random_zeroshot_ddpm",
                label="Random Search + Zero-shot + DDPM",
                optimizer=OptimizerType.RANDOM,
                prompt_strategy=PromptStrategyType.ZERO_SHOT,
                scheduler=SchedulerType.DDPM,
            ),
            ComparisonPreset(
                preset_id="random_neuro_ddim",
                label="Random Search + Нейросимволика + DDIM",
                optimizer=OptimizerType.RANDOM,
                prompt_strategy=PromptStrategyType.NEUROSYMBOLIC,
                scheduler=SchedulerType.DDIM,
            ),
            ComparisonPreset(
                preset_id="ga_neuro_ddim",
                label="Genetic Algorithm + Нейросимволика + DDIM",
                optimizer=OptimizerType.GA,
                prompt_strategy=PromptStrategyType.NEUROSYMBOLIC,
                scheduler=SchedulerType.DDIM,
            ),
            ComparisonPreset(
                preset_id="pso_neuro_ddim",
                label="Hybrid PSO + Нейросимволика + DDIM",
                optimizer=OptimizerType.PSO,
                prompt_strategy=PromptStrategyType.NEUROSYMBOLIC,
                scheduler=SchedulerType.DDIM,
            ),
        ]


class ExperimentRunner:
    """Оркестратор запуска одиночных, повторных и сравнительных вычислительных экспериментов."""

    def __init__(
        self,
        observers: Optional[List[BaseExperimentObserver]] = None,
    ) -> None:
        self._observers: List[BaseExperimentObserver] = (
            list(observers) if observers is not None else []
        )

    def add_observer(self, observer: BaseExperimentObserver) -> None:
        self._observers.append(observer)

    @staticmethod
    def _compute_macro_metrics(
        config: ExperimentConfig,
        baseline_fitness: float,
        history: List[IterationLog],
        override_std: Optional[float] = None,
    ) -> MacroMetrics:
        """Вычисляет макро-метрики Уровня 2: ΔF, N_conv, T_total, σ_F, mean LPIPS."""
        final_fitness = history[-1].global_best_fitness if history else baseline_fitness

        if baseline_fitness > 0.0:
            relative_gain_pct = round(
                ((final_fitness - baseline_fitness) / baseline_fitness) * 100.0, 2
            )
        else:
            relative_gain_pct = 0.0

        convergence_iteration: Optional[int] = None
        for log in history:
            if log.global_best_fitness >= config.target_fitness:
                convergence_iteration = log.iteration
                break

        total_time_sec = round(sum(log.iteration_time_sec for log in history), 3)

        if override_std is not None:
            fitness_std = round(override_std, 4)
        elif history:
            final_pop_scores = [
                p.current_evaluation.fitness for p in history[-1].particles
            ]
            fitness_std = _sample_std(final_pop_scores)
        else:
            fitness_std = 0.0

        lpips_values = [
            p.current_evaluation.lpips_score
            for log in history
            for p in log.particles
        ]
        mean_lpips = (
            round(sum(lpips_values) / len(lpips_values), 4) if lpips_values else 0.0
        )

        return MacroMetrics(
            baseline_fitness=round(baseline_fitness, 4),
            final_fitness=round(final_fitness, 4),
            relative_gain_pct=relative_gain_pct,
            convergence_iteration=convergence_iteration,
            total_time_sec=total_time_sec,
            fitness_std=fitness_std,
            mean_lpips=mean_lpips,
        )

    def run_experiment(
        self,
        config: ExperimentConfig,
        seed: Optional[int] = None,
        experiment_id: Optional[str] = None,
    ) -> ExperimentResult:
        """Выполняет один полный цикл оптимизации (для прикладной вкладки 'Оптимизация')."""
        effective_seed = config.seed if seed is None else seed
        run_config = (
            _clone_config(config, seed=effective_seed)
            if effective_seed != config.seed
            else _clone_config(config)
        )
        exp_id = experiment_id or f"exp_{uuid.uuid4().hex[:8]}"

        facade, optimizer = ExperimentFactory.build(run_config, seed=effective_seed)

        for observer in self._observers:
            observer.on_experiment_start(exp_id, run_config)

        # Оценка сырого бейзлайна F_0 с последующим сбросом RNG для чистоты траектории
        baseline_eval = facade.evaluate_baseline()
        facade.reset(seed=effective_seed)

        init_log = optimizer.initialize()
        for observer in self._observers:
            observer.on_iteration(exp_id, init_log)

        for _ in range(run_config.max_iterations):
            step_log = optimizer.step()
            for observer in self._observers:
                observer.on_iteration(exp_id, step_log)

        macro_metrics = self._compute_macro_metrics(
            config=run_config,
            baseline_fitness=baseline_eval.fitness,
            history=optimizer.history,
        )

        result = ExperimentResult(
            experiment_id=exp_id,
            config=run_config,
            macro_metrics=macro_metrics,
            history=optimizer.history,
        )

        for observer in self._observers:
            observer.on_experiment_end(result)

        return result

    def run_repeated_experiment(
        self,
        config: ExperimentConfig,
        num_runs: int = 5,
    ) -> Dict[str, Any]:
        """
        Проводит серию из M независимых прогонов с разными seed для оценки
        робастности алгоритма (истинное σ_F по финальным значениям F_best).
        """
        if num_runs < 1:
            raise ValueError("num_runs должно быть не менее 1.")

        base_seed = config.seed if config.seed is not None else 42
        runs: List[ExperimentResult] = []
        for run_idx in range(num_runs):
            runs.append(self.run_experiment(config, seed=base_seed + run_idx))

        final_scores = [r.macro_metrics.final_fitness for r in runs]
        gains = [r.macro_metrics.relative_gain_pct for r in runs]
        times = [r.macro_metrics.total_time_sec for r in runs]
        lpips_vals = [r.macro_metrics.mean_lpips for r in runs]
        conv_iters = [
            r.macro_metrics.convergence_iteration
            for r in runs
            if r.macro_metrics.convergence_iteration is not None
        ]

        # Репрезентативный прогон (первый) с обновленным межпрогонным σ_F
        representative = runs[0]
        true_std = _sample_std(final_scores) if num_runs > 1 else representative.macro_metrics.fitness_std

        return {
            "num_runs": num_runs,
            "mean_final_fitness": round(sum(final_scores) / num_runs, 4),
            "fitness_std": true_std,
            "mean_relative_gain_pct": round(sum(gains) / num_runs, 2),
            "mean_total_time_sec": round(sum(times) / num_runs, 3),
            "mean_lpips": round(sum(lpips_vals) / num_runs, 4),
            "convergence_rate_pct": round((len(conv_iters) / num_runs) * 100.0, 1),
            "mean_convergence_iteration": (
                round(sum(conv_iters) / len(conv_iters), 2) if conv_iters else None
            ),
            "representative_run": representative.to_dict(),
        }

    def run_comparison(
        self,
        base_config: ExperimentConfig,
        presets: Optional[List[ComparisonPreset]] = None,
        num_runs: int = 3,
    ) -> Dict[str, Any]:
        """
        Пакетный сравнительный анализ конфигураций на одном и том же запросе T_0
        (для исследовательской вкладки 'Сравнение методов' и экспорта JSON).
        """
        if num_runs < 1:
            raise ValueError("num_runs должно быть не менее 1.")

        active_presets = (
            presets if presets is not None else ExperimentFactory.default_comparison_presets()
        )
        comparison_id = f"cmp_{uuid.uuid4().hex[:8]}"

        entries: List[Dict[str, Any]] = []
        summary_table: List[Dict[str, Any]] = []

        for preset in active_presets:
            preset_config = _clone_config(
                base_config,
                optimizer=preset.optimizer,
                prompt_strategy=preset.prompt_strategy,
                scheduler=preset.scheduler,
            )
            stats = self.run_repeated_experiment(preset_config, num_runs=num_runs)

            rep = stats["representative_run"]
            curve = [
                {
                    "iteration": item["iteration"],
                    "global_best_fitness": item["global_best_fitness"],
                    "mean_fitness": item["mean_fitness"],
                }
                for item in rep["history"]
            ]

            entries.append(
                {
                    "preset_id": preset.preset_id,
                    "label": preset.label,
                    "optimizer": preset.optimizer.value,
                    "prompt_strategy": preset.prompt_strategy.value,
                    "scheduler": preset.scheduler.value,
                    "aggregated": {
                        "mean_final_fitness": stats["mean_final_fitness"],
                        "fitness_std": stats["fitness_std"],
                        "mean_relative_gain_pct": stats["mean_relative_gain_pct"],
                        "mean_total_time_sec": stats["mean_total_time_sec"],
                        "mean_lpips": stats["mean_lpips"],
                        "convergence_rate_pct": stats["convergence_rate_pct"],
                        "mean_convergence_iteration": stats["mean_convergence_iteration"],
                    },
                    "convergence_curve": curve,
                    "representative_run": rep,
                }
            )

            summary_table.append(
                {
                    "preset_id": preset.preset_id,
                    "label": preset.label,
                    "mean_final_fitness": stats["mean_final_fitness"],
                    "mean_relative_gain_pct": stats["mean_relative_gain_pct"],
                    "mean_convergence_iteration": stats["mean_convergence_iteration"],
                    "mean_total_time_sec": stats["mean_total_time_sec"],
                    "fitness_std": stats["fitness_std"],
                    "mean_lpips": stats["mean_lpips"],
                }
            )

        best_entry = max(summary_table, key=lambda row: row["mean_final_fitness"])

        return {
            "comparison_id": comparison_id,
            "core_prompt": base_config.core_prompt,
            "num_runs": num_runs,
            "target_fitness": base_config.target_fitness,
            "winner_preset_id": best_entry["preset_id"],
            "winner_label": best_entry["label"],
            "summary_table": summary_table,
            "presets": entries,
        }


def export_json(payload: Any, filepath: str | Path) -> Path:
    """Сохраняет ExperimentResult или словарь сравнения в форматированный UTF-8 JSON."""
    target = Path(filepath)
    target.parent.mkdir(parents=True, exist_ok=True)
    serializable = payload.to_dict() if hasattr(payload, "to_dict") else payload
    with target.open("w", encoding="utf-8") as fh:
        json.dump(serializable, fh, ensure_ascii=False, indent=2)
    return target
