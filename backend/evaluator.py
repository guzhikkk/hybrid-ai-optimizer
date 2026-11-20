from __future__ import annotations

from typing import List, Optional, Tuple

from adapters import BaseDiffusionAdapter, create_diffusion_adapter
from models import EvaluationBreakdown, ExperimentConfig, SchedulerType
from prompt_manager import (
    BasePromptStrategy,
    DecodedPrompt,
    create_prompt_strategy,
    tokenize_text,
)


class EvaluationFacade:
    """
    Фасад оценки качества гибридного ИИ-конвейера (паттерн Facade).
    Скрывает декодирование промпта LLM, запуск диффузионного сэмплера и расчет метрик
    за единым интерфейсом evaluate(position) -> EvaluationBreakdown.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        prompt_strategy: Optional[BasePromptStrategy] = None,
        adapter: Optional[BaseDiffusionAdapter] = None,
    ) -> None:
        self.config = config
        self.prompt_strategy = (
            prompt_strategy
            if prompt_strategy is not None
            else create_prompt_strategy(config)
        )
        self.adapter = (
            adapter if adapter is not None else create_diffusion_adapter(config)
        )
        self._eval_count: int = 0
        self._last_evaluation: Optional[EvaluationBreakdown] = None

    @property
    def eval_count(self) -> int:
        """Счетчик обращений к диффузионному конвейеру в текущем прогоне."""
        return self._eval_count

    @property
    def last_evaluation(self) -> Optional[EvaluationBreakdown]:
        """Результат последнего вычисления оценки."""
        return self._last_evaluation

    @property
    def bounds(self) -> List[Tuple[float, float]]:
        """Границы пространства поиска X_i из конфигурации эксперимента."""
        return self.config.get_search_bounds()

    def reset(self, seed: Optional[int] = None) -> None:
        """Сбрасывает историю стратегии промптов, состояние адаптера и счетчики."""
        self._eval_count = 0
        self._last_evaluation = None
        self.prompt_strategy.reset()
        self.adapter.reset(seed=seed if seed is not None else self.config.seed)

    def _compute_integral_fitness(
        self, clip_score: float, vis_score: float, structural_penalty: float
    ) -> Tuple[float, float]:
        """
        Вычисляет интегральную фитнес-функцию F(X) и функцию потерь J(X) = 1 - F(X)
        по формуле из Главы 2.3:
            F = w_clip * S_CLIP + w_vis * S_Vis - w_penalty * P_struct
        """
        w = self.config.metric_weights
        raw_fitness = (
            w.w_clip * clip_score
            + w.w_vis * vis_score
            - w.w_penalty * structural_penalty
        )
        fitness = round(max(0.0, min(1.0, raw_fitness)), 4)
        cost = round(1.0 - fitness, 4)
        return fitness, cost

    def evaluate(self, position: List[float]) -> EvaluationBreakdown:
        """Полная оценка вектора позиции частицы X_i с обновлением обратной связи LLM."""
        decoded = self.prompt_strategy.decode(position)
        adapter_out = self.adapter.evaluate_generation(
            decoded=decoded,
            position=position,
            core_prompt=self.config.core_prompt,
            is_baseline=False,
        )

        fitness, cost = self._compute_integral_fitness(
            clip_score=adapter_out.clip_score,
            vis_score=adapter_out.vis_score,
            structural_penalty=decoded.structural_penalty,
        )

        # Обратная связь для эволюционных стратегий LLM (In-Context Learning)
        self.prompt_strategy.update_history(decoded.prompt, fitness)

        breakdown = EvaluationBreakdown(
            prompt=decoded.prompt,
            guidance_scale=decoded.guidance_scale,
            steps=decoded.steps,
            clip_score=adapter_out.clip_score,
            vis_score=adapter_out.vis_score,
            structural_penalty=decoded.structural_penalty,
            lpips_score=adapter_out.lpips_score,
            fitness=fitness,
            cost=cost,
            eval_time_sec=adapter_out.eval_time_sec,
            image_url=adapter_out.image_url,
        )

        self._eval_count += 1
        self._last_evaluation = breakdown
        return breakdown

    def cost_function(self, position: List[float]) -> float:
        """Совместимая обертка для алгоритмов минимизации: возвращает J(X_i) = 1 - F(X_i)."""
        return self.evaluate(position).cost

    def evaluate_baseline(self) -> EvaluationBreakdown:
        """
        Вычисляет базовую оценку F_0 на «сыром» запросе пользователя T_0
        без оптимизации промпта и гиперпараметров (не влияет на eval_count и историю ICL).
        """
        baseline_steps = 50 if self.config.scheduler == SchedulerType.DDPM else 25
        baseline_g = 7.5
        num_slots = len(self.config.slot_dictionaries)
        neutral_position = (
            [baseline_g, float(baseline_steps)]
            + [1.0] * num_slots
            + [0.0] * num_slots
        )

        raw_decoded = DecodedPrompt(
            prompt=self.config.core_prompt,
            guidance_scale=baseline_g,
            steps=baseline_steps,
            token_count=len(tokenize_text(self.config.core_prompt)),
            core_retention_ratio=1.0,
            structural_penalty=0.0,
            strategy=self.config.prompt_strategy,
        )

        adapter_out = self.adapter.evaluate_generation(
            decoded=raw_decoded,
            position=neutral_position,
            core_prompt=self.config.core_prompt,
            is_baseline=True,
        )
        fitness, cost = self._compute_integral_fitness(
            clip_score=adapter_out.clip_score,
            vis_score=adapter_out.vis_score,
            structural_penalty=0.0,
        )

        return EvaluationBreakdown(
            prompt=raw_decoded.prompt,
            guidance_scale=baseline_g,
            steps=baseline_steps,
            clip_score=adapter_out.clip_score,
            vis_score=adapter_out.vis_score,
            structural_penalty=0.0,
            lpips_score=adapter_out.lpips_score,
            fitness=fitness,
            cost=cost,
            eval_time_sec=adapter_out.eval_time_sec,
            image_url=adapter_out.image_url,
        )
