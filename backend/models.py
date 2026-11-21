from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class OptimizerType(str, Enum):
    """Поддерживаемые алгоритмы оптимизации (Узел 1, FC-1.1)."""

    RANDOM = "random"
    GA = "ga"
    PSO = "pso"


class PromptStrategyType(str, Enum):
    """Стратегии управления промпт-пространством LLM (Узел 2, FC-2.1)."""

    ZERO_SHOT = "zero_shot"
    IN_CONTEXT = "in_context"
    NEUROSYMBOLIC = "neurosymbolic"


class SchedulerType(str, Enum):
    """Планировщики сэмплирования латентной диффузии (Узел 3, FC-2.2)."""

    DDPM = "ddpm"
    DDIM = "ddim"


class BackendMode(str, Enum):
    """Режим исполнения нейросетевых адаптеров."""

    SURROGATE = "surrogate"      # Быстрая мат. модель для Render, демо и pytest
    API = "api"                  # Внешние Inference API (HuggingFace / LLM)
    COLAB_LOCAL = "colab_local"  # Локальный инференс Diffusers + CLIP на GPU


def default_slot_dictionaries() -> Dict[str, List[str]]:
    """
    Возвращает свежую копию словаря слотов нейросимволического шаблона.
    Используется как default_factory, чтобы исключить разделение мутабельных списков.
    """
    return {       
        "style": [
            "masterpiece editorial photography, rich colors",
            "cinematic film still, shallow depth of field",
            "award-winning digital art, balanced composition",
            "fine art studio visual, harmonious tones",
        ],
        "lighting": [
            "cinematic volumetric lighting, soft shadows",
            "soft natural studio rim light",
            "warm golden hour sunlight glow",
            "dramatic atmospheric lighting, crisp highlights",
        ],
        "detail": [
            "sharp focus, intricate lifelike textures",
            "ultra detailed surfaces, natural color grading",
            "crisp details, subtle depth, clean framing",
            "high dynamic range, refined clarity",
        ],
    }


def _to_serializable(obj: Any) -> Any:
    """Рекурсивно преобразует dataclass, Enum и коллекции в чистый JSON-совместимый вид."""
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj):
        return {f.name: _to_serializable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(item) for item in obj]
    return obj


@dataclass
class MetricWeights:
    """Весовые коэффициенты композитной целевой функции F(X) и штрафа P_struct."""

    w_clip: float = 0.55
    w_vis: float = 0.45
    w_penalty: float = 0.50
    alpha: float = 0.60  # Вес потери ключевых токенов ядра
    beta: float = 0.40   # Вес превышения лимита токенов CLIP

    def __post_init__(self) -> None:
        for name in ("w_clip", "w_vis", "w_penalty", "alpha", "beta"):
            val = getattr(self, name)
            if val < 0.0:
                raise ValueError(f"Вес '{name}' не может быть отрицательным (получено {val}).")

        if not math.isclose(self.w_clip + self.w_vis, 1.0, abs_tol=1e-6):
            raise ValueError(
                f"Сумма w_clip + w_vis должна равняться 1.0 (получено {self.w_clip + self.w_vis:.4f})."
            )
        if not math.isclose(self.alpha + self.beta, 1.0, abs_tol=1e-6):
            raise ValueError(
                f"Сумма alpha + beta должна равняться 1.0 (получено {self.alpha + self.beta:.4f})."
            )

    def to_dict(self) -> Dict[str, Any]:
        return _to_serializable(self)


@dataclass
class ExperimentConfig:
    """Полная конфигурация запуска эксперимента."""

    core_prompt: str = "3D mascot raccoon wearing a green hoodie and backpack"
    optimizer: OptimizerType = OptimizerType.PSO
    prompt_strategy: PromptStrategyType = PromptStrategyType.NEUROSYMBOLIC
    scheduler: SchedulerType = SchedulerType.DDIM
    backend_mode: BackendMode = BackendMode.SURROGATE

    # Общие параметры бюджета оптимизации
    num_particles: int = 8
    max_iterations: int = 15
    target_fitness: float = 0.82
    max_tokens: int = 77
    seed: Optional[int] = 42

    # Гиперпараметры метода роя частиц (PSO)
    pso_w: float = 0.72
    pso_c1: float = 1.49
    pso_c2: float = 1.49

    # Гиперпараметры генетического алгоритма (GA)
    ga_elitism: int = 2
    ga_crossover_rate: float = 0.80
    ga_mutation_rate: float = 0.15

    # Веса метрик и слоты нейросимволического шаблона
    metric_weights: MetricWeights = field(default_factory=MetricWeights)
    slot_dictionaries: Dict[str, List[str]] = field(default_factory=default_slot_dictionaries)

    def __post_init__(self) -> None:
        # Автоматическое приведение строковых значений к Enum
        if isinstance(self.optimizer, str):
            self.optimizer = OptimizerType(self.optimizer.lower())
        if isinstance(self.prompt_strategy, str):
            self.prompt_strategy = PromptStrategyType(self.prompt_strategy.lower())
        if isinstance(self.scheduler, str):
            self.scheduler = SchedulerType(self.scheduler.lower())
        if isinstance(self.backend_mode, str):
            self.backend_mode = BackendMode(self.backend_mode.lower())
        if isinstance(self.metric_weights, dict):
            self.metric_weights = MetricWeights(**self.metric_weights)

        self.core_prompt = self.core_prompt.strip()
        if not self.core_prompt:
            raise ValueError("core_prompt не может быть пустым.")
        if self.num_particles < 2:
            raise ValueError("num_particles должно быть не менее 2.")
        if self.max_iterations < 1:
            raise ValueError("max_iterations должно быть не менее 1.")
        if not (0.0 < self.target_fitness <= 1.0):
            raise ValueError("target_fitness должно лежать в интервале (0.0, 1.0].")
        if self.max_tokens < 10:
            raise ValueError("max_tokens должно быть не менее 10.")
        if not (0 <= self.ga_elitism < self.num_particles):
            raise ValueError("ga_elitism должно удовлетворять 0 <= ga_elitism < num_particles.")
        if not (0.0 <= self.ga_crossover_rate <= 1.0):
            raise ValueError("ga_crossover_rate должно лежать в [0.0, 1.0].")
        if not (0.0 <= self.ga_mutation_rate <= 1.0):
            raise ValueError("ga_mutation_rate должно лежать в [0.0, 1.0].")
        if not self.slot_dictionaries:
            raise ValueError("slot_dictionaries не может быть пустым.")
        for slot_name, options in self.slot_dictionaries.items():
            if not options:
                raise ValueError(f"Список вариантов для слота '{slot_name}' не может быть пустым.")

    def get_search_bounds(self) -> List[Tuple[float, float]]:
        """
        Возвращает границы D-мерного вектора позиции частицы X_i:
        [guidance_scale, steps, w_1..w_K, z_1..z_K]
        """
        num_slots = len(self.slot_dictionaries)
        bounds: List[Tuple[float, float]] = [
            (3.0, 15.0),   # index 0: Guidance Scale (g)
            (10.0, 50.0),  # index 1: Шаги денойзинга (s)
        ]
        bounds.extend([(0.5, 1.5)] * num_slots)  # Веса слотов w_k
        bounds.extend([(0.0, 1.0)] * num_slots)  # Селекторы дескрипторов z_k
        return bounds

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExperimentConfig:
        """Безопасное создание конфигурации из JSON-словаря (например, тела запроса Flask)."""
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys and v is not None}
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        return _to_serializable(self)


@dataclass
class EvaluationBreakdown:
    """Детальный отчет оценки одной точки X_i (метрики Уровня 1)."""

    prompt: str
    guidance_scale: float
    steps: int
    clip_score: float
    vis_score: float
    structural_penalty: float
    lpips_score: float
    fitness: float
    cost: float
    eval_time_sec: float
    image_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _to_serializable(self)


@dataclass
class ParticleSnapshot:
    """Снимок состояния одной частицы/особи на конкретной итерации (для визуализации)."""

    particle_id: int
    position: List[float]
    velocity: List[float]
    best_position: List[float]
    best_fitness: float
    current_evaluation: EvaluationBreakdown

    def to_dict(self) -> Dict[str, Any]:
        return _to_serializable(self)


@dataclass
class IterationLog:
    """Лог одной итерации алгоритма оптимизации."""

    iteration: int
    global_best_fitness: float
    global_best_cost: float
    global_best_position: List[float]
    global_best_evaluation: EvaluationBreakdown
    mean_fitness: float
    iteration_time_sec: float
    particles: List[ParticleSnapshot]

    def to_dict(self) -> Dict[str, Any]:
        return _to_serializable(self)


@dataclass
class MacroMetrics:
    """Итоговые макро-метрики эффективности алгоритма (метрики Уровня 2)."""

    baseline_fitness: float
    final_fitness: float
    relative_gain_pct: float
    convergence_iteration: Optional[int]
    total_time_sec: float
    fitness_std: float
    mean_lpips: float

    def to_dict(self) -> Dict[str, Any]:
        return _to_serializable(self)


@dataclass
class ExperimentResult:
    """Полный результат эксперимента, экспортируемый в API и JSON для GitHub Pages."""

    experiment_id: str
    config: ExperimentConfig
    macro_metrics: MacroMetrics
    history: List[IterationLog]

    def to_dict(self) -> Dict[str, Any]:
        return _to_serializable(self)