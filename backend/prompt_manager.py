from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from models import (
    ExperimentConfig,
    MetricWeights,
    PromptStrategyType,
    default_slot_dictionaries,
)

# Сигнатура опционального адаптера внешней LLM: (system_prompt, user_prompt) -> text
LLMCallback = Callable[[str, str], str]

_TOKEN_PATTERN = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]+")


def tokenize_text(text: str) -> List[str]:
    """Извлекает нормализованные словесные токены (поддерживает EN и RU)."""
    return _TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class DecodedPrompt:
    """Результат декодирования вектора частицы X_i в текстовый промпт и параметры диффузии."""

    prompt: str
    guidance_scale: float
    steps: int
    token_count: int
    core_retention_ratio: float
    structural_penalty: float
    strategy: PromptStrategyType


class BasePromptStrategy(ABC):
    """Абстрактная стратегия управления промпт-пространством LLM (паттерн Strategy)."""

    strategy_type: PromptStrategyType

    def __init__(
        self,
        core_prompt: str,
        slot_dictionaries: Optional[Dict[str, List[str]]] = None,
        max_tokens: int = 77,
        metric_weights: Optional[MetricWeights] = None,
        llm_callback: Optional[LLMCallback] = None,
    ) -> None:
        self.core_prompt = core_prompt.strip()
        if not self.core_prompt:
            raise ValueError("core_prompt не может быть пустым.")

        self.slot_dictionaries = (
            slot_dictionaries if slot_dictionaries is not None else default_slot_dictionaries()
        )
        if not self.slot_dictionaries:
            raise ValueError("slot_dictionaries не может быть пустым.")

        self.slot_names: List[str] = list(self.slot_dictionaries.keys())
        for name, options in self.slot_dictionaries.items():
            if not options:
                raise ValueError(f"Слот '{name}' не содержит ни одного дескриптора.")

        self.max_tokens = max_tokens
        self.metric_weights = metric_weights if metric_weights is not None else MetricWeights()
        self.llm_callback = llm_callback
        self._core_tokens = set(tokenize_text(self.core_prompt))

    @property
    def expected_vector_dim(self) -> int:
        """Ожидаемая размерность вектора частицы: [g, s, w_1..w_K, z_1..z_K]."""
        return 2 + 2 * len(self.slot_names)

    def _validate_and_unpack(
        self, position: List[float]
    ) -> Tuple[float, int, List[float], List[float]]:
        """Валидирует вектор X_i и извлекает (guidance_scale, steps, weights, selectors)."""
        if len(position) < self.expected_vector_dim:
            raise ValueError(
                f"Ожидался вектор размерности не менее {self.expected_vector_dim}, "
                f"получено {len(position)}."
            )

        guidance_scale = round(max(1.0, min(20.0, float(position[0]))), 2)
        steps = max(1, min(100, int(round(float(position[1])))))

        num_slots = len(self.slot_names)
        weights = [
            round(max(0.5, min(1.5, float(w))), 2)
            for w in position[2 : 2 + num_slots]
        ]
        selectors = [
            max(0.0, min(1.0, float(z)))
            for z in position[2 + num_slots : 2 + 2 * num_slots]
        ]
        return guidance_scale, steps, weights, selectors

    def _select_descriptor(self, slot_name: str, selector: float) -> str:
        """Отображает непрерывный параметр z_k in [0, 1] в конкретный дескриптор слота."""
        options = self.slot_dictionaries[slot_name]
        idx = min(int(selector * len(options)), len(options) - 1)
        return options[max(0, idx)]

    def calculate_core_retention(self, prompt: str) -> float:
        """Доля ключевых токенов ядра T_0, сохранившихся в сгенерированном промпте."""
        if not self._core_tokens:
            return 1.0
        prompt_tokens = set(tokenize_text(prompt))
        retained = len(self._core_tokens.intersection(prompt_tokens))
        return round(retained / len(self._core_tokens), 4)

    def calculate_structural_penalty(self, prompt: str) -> float:
        """
        Расчет структурного штрафа P_struct in [0, 1] по формуле из Главы 2.3:
        P_struct = alpha * (1 - retention_ratio) + beta * max(0, (|P| - L_max) / L_max)
        """
        retention_ratio = self.calculate_core_retention(prompt)
        retention_loss = 1.0 - retention_ratio

        token_count = len(tokenize_text(prompt))
        overflow_loss = max(0.0, (token_count - self.max_tokens) / float(self.max_tokens))

        raw_penalty = (
            self.metric_weights.alpha * retention_loss
            + self.metric_weights.beta * overflow_loss
        )
        return round(min(1.0, max(0.0, raw_penalty)), 4)

    def update_history(self, prompt: str, fitness: float) -> None:
        """Хук обратной связи для стратегий, использующих историю оценок (ICL)."""
        return None

    def reset(self) -> None:
        """Сброс внутреннего состояния стратегии перед новым прогоном эксперимента."""
        return None

    @abstractmethod
    def _build_prompt_text(
        self, weights: List[float], selectors: List[float]
    ) -> str:
        """Формирует итоговую текстовую строку промпта согласно выбранной стратегии."""

    def decode(self, position: List[float]) -> DecodedPrompt:
        """Полное декодирование вектора частицы X_i в промпт, параметры и штраф."""
        guidance_scale, steps, weights, selectors = self._validate_and_unpack(position)
        prompt_text = self._build_prompt_text(weights, selectors)

        token_count = len(tokenize_text(prompt_text))
        retention_ratio = self.calculate_core_retention(prompt_text)
        penalty = self.calculate_structural_penalty(prompt_text)

        return DecodedPrompt(
            prompt=prompt_text,
            guidance_scale=guidance_scale,
            steps=steps,
            token_count=token_count,
            core_retention_ratio=retention_ratio,
            structural_penalty=penalty,
            strategy=self.strategy_type,
        )


class ZeroShotPromptStrategy(BasePromptStrategy):
    """
    Стратегия 1 (Бейзлайн): свободная Zero-shot генерация промпта силами LLM.
    Без жестких символьных ограничений склонна к многословности (> 77 токенов)
    и частичной потере исходных сущностей запроса.
    """

    strategy_type = PromptStrategyType.ZERO_SHOT

    _VERBOSE_INTROS: Tuple[str, ...] = (
        "An extraordinarily elaborate, breathtaking digital illustration depicting",
        "A richly atmospheric, cinematic fine-art composition showcasing",
        "An ultra-imaginative, museum-grade conceptual scene representing",
    )
    _VERBOSE_FILLERS: Tuple[str, ...] = (
        "surrounded by sweeping atmospheric haze, intricate storytelling elements, "
        "dramatic interplay of reflections, subtle particle dust floating in the air, "
        "hyper-complex background architecture stretching far into the misty horizon, "
        "layered artistic brushwork, vivid chromatic depth, and expressive composition "
        "crafted with obsessive attention to micro-details across every surface, "
        "masterpiece quality, award-winning gallery artwork, dramatic camera lens bokeh",
        "featuring rich environmental storytelling, complex visual metaphors, "
        "dynamic shadows cascading over textured surfaces, subtle color grading inspired "
        "by classic cinema, ultra-crisp depth of field, ambient atmospheric glow, "
        "trending on art platforms, polished visual hierarchy, dramatic perspective",
        "rendered with expansive background scenery, lush environmental accents, "
        "multi-layered lighting gradients, expressive contrast, sweeping cinematic mood, "
        "intricate peripheral details, harmonious color palette, artistic framing",
    )

    def _build_prompt_text(
        self, weights: List[float], selectors: List[float]
    ) -> str:
        if self.llm_callback is not None:
            system_msg = (
                "You are a creative prompt writer for image generation. "
                "Expand and enrich the user request into a vivid descriptive paragraph."
            )
            return self.llm_callback(system_msg, self.core_prompt).strip()

        # Детерминированная оффлайн-модель Zero-shot поведения LLM
        avg_z = sum(selectors) / len(selectors)
        avg_w = sum(weights) / len(weights)

        intro_idx = min(int(avg_z * len(self._VERBOSE_INTROS)), len(self._VERBOSE_INTROS) - 1)
        filler_idx = min(
            int(((avg_w - 0.5) / 1.0) * len(self._VERBOSE_FILLERS)),
            len(self._VERBOSE_FILLERS) - 1,
        )
        filler_idx = max(0, filler_idx)

        # В ~40% пространства поиска Zero-shot LLM перефразирует/теряет часть ядра
        core_words = self.core_prompt.split()
        if avg_z > 0.60 and len(core_words) > 3:
            keep_count = max(2, int(len(core_words) * 0.55))
            subject_fragment = " ".join(core_words[:keep_count]) + " in a stylized fantasy world"
        else:
            subject_fragment = self.core_prompt

        descriptors = [
            self._select_descriptor(slot_name, selectors[i])
            for i, slot_name in enumerate(self.slot_names)
        ]

        # При высоком avg_w добавляется длинное художественное описание (> 77 токенов)
        if avg_w >= 0.85:
            extra_tail = self._VERBOSE_FILLERS[0]
        else:
            extra_tail = self._VERBOSE_FILLERS[filler_idx]

        return (
            f"{self._VERBOSE_INTROS[intro_idx]} {subject_fragment}, "
            f"{', '.join(descriptors)}, {extra_tail}"
        )


class InContextPromptStrategy(BasePromptStrategy):
    """
    Стратегия 2: эволюционный промптинг в контексте (In-Context Learning, ICL).
    Использует историю лучших пар (промпт, оценка F), но при накоплении итераций
    подвержена умеренному семантическому дрейфу (наслоению стилевых штампов).
    """

    strategy_type = PromptStrategyType.IN_CONTEXT

    _STYLE_CLICHES: Tuple[str, ...] = (
        "trending on artstation, dramatic neon rim glow, hyper-stylized atmosphere",
        "volumetric cinematic fog, award-winning octane aesthetic, intricate surreal accents",
        "unreal engine showcase, polished studio reflections, expressive dramatic contrast",
    )

    def __init__(
        self,
        core_prompt: str,
        slot_dictionaries: Optional[Dict[str, List[str]]] = None,
        max_tokens: int = 77,
        metric_weights: Optional[MetricWeights] = None,
        llm_callback: Optional[LLMCallback] = None,
        max_history: int = 5,
    ) -> None:
        super().__init__(
            core_prompt=core_prompt,
            slot_dictionaries=slot_dictionaries,
            max_tokens=max_tokens,
            metric_weights=metric_weights,
            llm_callback=llm_callback,
        )
        self.max_history = max_history
        self._history: List[Tuple[str, float]] = []
        self._step_counter: int = 0

    def update_history(self, prompt: str, fitness: float) -> None:
        """Сохраняет лучшие примеры (prompt, fitness) в контекстном окне стратегии."""
        self._history.append((prompt, fitness))
        self._history.sort(key=lambda item: item[1], reverse=True)
        self._history = self._history[: self.max_history]

    def reset(self) -> None:
        self._history.clear()
        self._step_counter = 0

    @property
    def history(self) -> List[Tuple[str, float]]:
        return list(self._history)

    def _build_prompt_text(
        self, weights: List[float], selectors: List[float]
    ) -> str:
        self._step_counter += 1

        if self.llm_callback is not None:
            examples = "\n".join(
                f"- Score {score:.3f}: {text}" for text, score in self._history[:3]
            )
            system_msg = (
                "You are an evolutionary prompt optimizer. Given past prompts and their quality "
                "scores, synthesize an improved concise prompt under 75 words."
            )
            user_msg = (
                f"Original task: {self.core_prompt}\n"
                f"Top previous attempts:\n{examples or 'None yet'}\n"
                "Generate one improved prompt:"
            )
            return self.llm_callback(system_msg, user_msg).strip()

        # Детерминированная оффлайн-модель ICL поведения
        descriptors = [
            self._select_descriptor(slot_name, selectors[i])
            for i, slot_name in enumerate(self.slot_names)
        ]

        avg_z = sum(selectors) / len(selectors)
        core_words = self.core_prompt.split()

        # Умеренный семантический дрейф на поздних шагах или крайних z (~15% случаев)
        if self._history and avg_z > 0.82 and len(core_words) > 3:
            subject_part = " ".join(core_words[:-1])
        else:
            subject_part = self.core_prompt

        cliche_idx = min(
            int(avg_z * len(self._STYLE_CLICHES)), len(self._STYLE_CLICHES) - 1
        )
        parts = [subject_part, *descriptors]
        if self._step_counter > 2 or self._history:
            parts.append(self._STYLE_CLICHES[cliche_idx])

        return ", ".join(parts)


class NeurosymbolicPromptStrategy(BasePromptStrategy):
    """
    Стратегия 3 (Предложенная): нейросимволическое управление LLM.
    Жестко фиксирует инвариантное ядро Core(T_0), параметризует слоты непрерывными
    весами внимания (descriptor:w_k) и строго соблюдает лимит max_tokens (P_struct = 0).
    """

    strategy_type = PromptStrategyType.NEUROSYMBOLIC

    def _build_prompt_text(
        self, weights: List[float], selectors: List[float]
    ) -> str:
        parts: List[str] = [self.core_prompt]
        current_tokens = len(tokenize_text(self.core_prompt))

        for idx, slot_name in enumerate(self.slot_names):
            descriptor = self._select_descriptor(slot_name, selectors[idx])
            weight = weights[idx]
            slot_fragment = f"({descriptor}:{weight:.2f})"
            fragment_tokens = len(tokenize_text(slot_fragment))

            # Символьный инвариант: не допускаем превышения контекстного окна CLIP
            if current_tokens + fragment_tokens <= self.max_tokens:
                parts.append(slot_fragment)
                current_tokens += fragment_tokens

        return ", ".join(parts)


def create_prompt_strategy(
    config: ExperimentConfig,
    llm_callback: Optional[LLMCallback] = None,
) -> BasePromptStrategy:
    """Фабричный метод создания стратегии управления LLM из конфигурации эксперимента."""
    strategy_map = {
        PromptStrategyType.ZERO_SHOT: ZeroShotPromptStrategy,
        PromptStrategyType.IN_CONTEXT: InContextPromptStrategy,
        PromptStrategyType.NEUROSYMBOLIC: NeurosymbolicPromptStrategy,
    }
    strategy_cls = strategy_map[config.prompt_strategy]
    return strategy_cls(
        core_prompt=config.core_prompt,
        slot_dictionaries=config.slot_dictionaries,
        max_tokens=config.max_tokens,
        metric_weights=config.metric_weights,
        llm_callback=llm_callback,
    )
