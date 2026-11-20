from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Добавляем backend/ в sys.path для запуска pytest из любой директории
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app import create_app
from evaluator import EvaluationFacade
from experiment_runner import ExperimentRunner, MetricsLoggerObserver
from models import (
    ExperimentConfig,
    MetricWeights,
    OptimizerType,
    PromptStrategyType,
    SchedulerType,
)
from optimizers import create_optimizer
from prompt_manager import (
    InContextPromptStrategy,
    NeurosymbolicPromptStrategy,
    ZeroShotPromptStrategy,
)


def test_config_validation_and_bounds() -> None:
    """Тест 1: Проверка дефолтных границ вектора X_i и валидации весов метрик."""
    cfg = ExperimentConfig()
    bounds = cfg.get_search_bounds()
    assert len(bounds) == 2 + 2 * len(cfg.slot_dictionaries)
    assert bounds[0] == (3.0, 15.0)
    assert bounds[1] == (10.0, 50.0)

    with pytest.raises(ValueError):
        MetricWeights(w_clip=0.7, w_vis=0.7)

    with pytest.raises(ValueError):
        ExperimentConfig(num_particles=1)


def test_neurosymbolic_strategy_preserves_core_without_penalty() -> None:
    """Тест 2: Нейросимволический шаблон сохраняет ядро и дает P_struct = 0."""
    strategy = NeurosymbolicPromptStrategy(
        core_prompt="3D mascot raccoon wearing a green hoodie"
    )
    vec = [7.5, 15.0, 1.1, 1.2, 1.0, 0.25, 0.5, 0.75]
    decoded = strategy.decode(vec)

    assert "3D mascot raccoon wearing a green hoodie" in decoded.prompt
    assert decoded.core_retention_ratio == 1.0
    assert decoded.structural_penalty == 0.0
    assert decoded.token_count <= strategy.max_tokens


def test_zeroshot_strategy_penalizes_core_loss_and_overflow() -> None:
    """Тест 3: Zero-shot генерация получает штраф при потере ядра и длинном описании."""
    strategy = ZeroShotPromptStrategy(
        core_prompt="3D mascot raccoon wearing a green hoodie and backpack"
    )
    vec = [8.0, 25.0, 1.2, 1.2, 1.1, 0.85, 0.80, 0.90]
    decoded = strategy.decode(vec)

    assert decoded.core_retention_ratio < 1.0
    assert decoded.structural_penalty > 0.0


def test_incontext_strategy_history_management() -> None:
    """Тест 4: In-Context стратегия сортирует историю по убыванию F и сбрасывается."""
    strategy = InContextPromptStrategy(
        core_prompt="cyberpunk neon tower at night", max_history=2
    )
    strategy.update_history("prompt_low", 0.55)
    strategy.update_history("prompt_high", 0.89)
    strategy.update_history("prompt_mid", 0.74)

    assert len(strategy.history) == 2
    assert strategy.history[0] == ("prompt_high", 0.89)
    assert strategy.history[1] == ("prompt_mid", 0.74)

    strategy.reset()
    assert strategy.history == []


def test_ddim_more_stable_than_ddpm() -> None:
    """Тест 5: Детерминированный DDIM дает ниже LPIPS и меньший шум отклика, чем DDPM."""
    vec = [7.8, 15.0, 1.1, 1.1, 1.1, 0.4, 0.5, 0.6]

    facade_ddim = EvaluationFacade(
        ExperimentConfig(scheduler=SchedulerType.DDIM, seed=42)
    )
    facade_ddpm = EvaluationFacade(
        ExperimentConfig(scheduler=SchedulerType.DDPM, seed=42)
    )

    ddim_runs = [facade_ddim.evaluate(vec) for _ in range(10)]
    ddpm_runs = [facade_ddpm.evaluate(vec) for _ in range(10)]

    def _std(vals: list[float]) -> float:
        mean = sum(vals) / len(vals)
        return math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))

    ddim_std = _std([r.fitness for r in ddim_runs])
    ddpm_std = _std([r.fitness for r in ddpm_runs])
    ddim_lpips = sum(r.lpips_score for r in ddim_runs) / len(ddim_runs)
    ddpm_lpips = sum(r.lpips_score for r in ddpm_runs) / len(ddpm_runs)

    assert ddim_std < ddpm_std
    assert ddim_lpips < ddpm_lpips


def test_pso_monotonic_fitness_and_strict_bounds() -> None:
    """Тест 6: PSO монотонно не ухудшает G_best и удерживает частицы в границах."""
    cfg = ExperimentConfig(
        optimizer=OptimizerType.PSO,
        num_particles=8,
        max_iterations=10,
        pso_w=1.4,
        seed=42,
    )
    facade = EvaluationFacade(cfg)
    opt = create_optimizer(cfg, facade.evaluate)
    bounds = cfg.get_search_bounds()

    log0 = opt.initialize()
    prev_best = log0.global_best_fitness

    for _ in range(cfg.max_iterations):
        log = opt.step()
        assert log.global_best_fitness >= prev_best
        prev_best = log.global_best_fitness
        for snapshot in log.particles:
            for idx, (low, high) in enumerate(bounds):
                assert low <= snapshot.position[idx] <= high


@pytest.mark.parametrize("opt_type", [OptimizerType.RANDOM, OptimizerType.GA])
def test_baseline_and_ga_optimizers_respect_bounds(opt_type: OptimizerType) -> None:
    """Тест 7: Random Search и GA соблюдают размер популяции, границы и монотонность."""
    cfg = ExperimentConfig(optimizer=opt_type, num_particles=6, max_iterations=6, seed=7)
    facade = EvaluationFacade(cfg)
    opt = create_optimizer(cfg, facade.evaluate)
    bounds = cfg.get_search_bounds()

    prev_best = opt.initialize().global_best_fitness
    for _ in range(cfg.max_iterations):
        log = opt.step()
        assert len(log.particles) == cfg.num_particles
        assert log.global_best_fitness >= prev_best
        prev_best = log.global_best_fitness
        for snapshot in log.particles:
            for idx, (low, high) in enumerate(bounds):
                assert low <= snapshot.position[idx] <= high


def test_comparison_runner_proves_pso_advantage() -> None:
    """Тест 8: Пакетный бенчмарк подтверждает превосходство PSO + Нейросимволика + DDIM."""
    observer = MetricsLoggerObserver()
    runner = ExperimentRunner(observers=[observer])
    cfg = ExperimentConfig(max_iterations=10, num_particles=8, target_fitness=0.90)

    cmp = runner.run_comparison(cfg, num_runs=2)
    rows = {row["preset_id"]: row for row in cmp["summary_table"]}

    baseline = rows["baseline_random_zeroshot_ddpm"]
    pso = rows["pso_neuro_ddim"]

    assert pso["mean_final_fitness"] > baseline["mean_final_fitness"]
    assert pso["mean_lpips"] < baseline["mean_lpips"]
    assert len(observer.records) == cfg.max_iterations + 1


def test_api_health_and_defaults() -> None:
    """Тест 9: Проверка служебных GET-эндпоинтов Flask API."""
    client = create_app().test_client()

    health_resp = client.get("/api/health")
    assert health_resp.status_code == 200
    assert health_resp.get_json()["status"] == "ok"

    defaults_resp = client.get("/api/config/defaults")
    assert defaults_resp.status_code == 200
    data = defaults_resp.get_json()
    assert "pso" in data["options"]["optimizers"]
    assert len(data["comparison_presets"]) == 4


def test_api_optimize_and_compare_endpoints() -> None:
    """Тест 10: Проверка POST /api/optimize, POST /api/compare и обработки ошибок."""
    client = create_app().test_client()

    opt_resp = client.post(
        "/api/optimize",
        json={
            "core_prompt": "futuristic racing car in neon rain",
            "optimizer": "pso",
            "prompt_strategy": "neurosymbolic",
            "scheduler": "ddim",
            "num_particles": 4,
            "max_iterations": 4,
        },
    )
    assert opt_resp.status_code == 200
    opt_data = opt_resp.get_json()
    assert len(opt_data["history"]) == 5
    assert "macro_metrics" in opt_data

    cmp_resp = client.post(
        "/api/compare",
        json={"num_particles": 4, "max_iterations": 4, "num_runs": 2},
    )
    assert cmp_resp.status_code == 200
    cmp_data = cmp_resp.get_json()
    assert len(cmp_data["summary_table"]) == 4

    bad_resp = client.post("/api/optimize", json={"optimizer": "invalid_algo"})
    assert bad_resp.status_code == 400
    assert "error" in bad_resp.get_json()