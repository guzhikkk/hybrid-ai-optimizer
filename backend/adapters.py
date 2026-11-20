from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import random
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

from models import BackendMode, ExperimentConfig, PromptStrategyType, SchedulerType
from prompt_manager import DecodedPrompt


@dataclass(frozen=True)
class AdapterOutput:
    """Сырые метрики оценки одной генерации, возвращаемые адаптером диффузии."""

    clip_score: float
    vis_score: float
    lpips_score: float
    eval_time_sec: float
    image_url: Optional[str] = None


class BaseDiffusionAdapter(ABC):
    """Абстрактный адаптер пайплайна латентной диффузии и метрик (паттерн Adapter)."""

    def __init__(
        self,
        scheduler: SchedulerType = SchedulerType.DDIM,
        max_tokens: int = 77,
        seed: Optional[int] = 42,
    ) -> None:
        self.scheduler = scheduler
        self.max_tokens = max_tokens
        self.seed = seed

    @abstractmethod
    def evaluate_generation(
        self,
        decoded: DecodedPrompt,
        position: List[float],
        core_prompt: str,
        is_baseline: bool = False,
    ) -> AdapterOutput:
        """Запускает генерацию (или модель среды) и возвращает метрики Уровня 1."""

    def reset(self, seed: Optional[int] = None) -> None:
        """Сбрасывает внутреннее состояние адаптера перед новым прогоном."""
        if seed is not None:
            self.seed = seed


class SurrogateDiffusionAdapter(BaseDiffusionAdapter):
    """
    Быстрая математическая модель среды «Латентная диффузия + CLIP/ImageReward».
    Используется для интерактивного демо на Render (без GPU) и модульных тестов pytest.
    Воспроизводит ключевые физические свойства:
      - детерминированность и быструю сходимость DDIM по сравнению со стохастичным DDPM;
      - деградацию CLIPScore при потере ядра запроса и переполнении окна 77 токенов;
      - оптимум по Guidance Scale g, шагам s, весам слотов w_k и дескрипторам z_k.
    """

    def __init__(
        self,
        scheduler: SchedulerType = SchedulerType.DDIM,
        max_tokens: int = 77,
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__(scheduler=scheduler, max_tokens=max_tokens, seed=seed)
        self._rng = random.Random(seed)
        self._prev_position: Optional[List[float]] = None

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed)
        self._rng = random.Random(self.seed)
        self._prev_position = None

    @staticmethod
    def _derive_prompt_optimum(
        core_prompt: str, num_slots: int
    ) -> Tuple[float, List[float], List[float]]:
        """Детерминированно вычисляет скрытый оптимум (opt_g, opt_w, opt_z) для данного запроса."""
        digest = hashlib.sha256(core_prompt.strip().lower().encode("utf-8")).digest()
        opt_g = 7.2 + (digest[0] / 255.0) * 1.8  # Диапазон оптимума Guidance Scale: [7.2; 9.0]
        opt_w = [
            1.0 + (digest[(1 + k) % len(digest)] / 255.0) * 0.30
            for k in range(num_slots)
        ]
        opt_z = [
            0.15 + (digest[(8 + k) % len(digest)] / 255.0) * 0.70
            for k in range(num_slots)
        ]
        return opt_g, opt_w, opt_z

    def _estimate_eval_time(self, decoded: DecodedPrompt, is_baseline: bool) -> float:
        """Расчетная вычислительная задержка одного шага конвейера на GPU (в секундах)."""
        if is_baseline:
            t_llm = 0.0
        elif decoded.strategy == PromptStrategyType.NEUROSYMBOLIC:
            t_llm = 0.08
        elif decoded.strategy == PromptStrategyType.ZERO_SHOT:
            t_llm = 0.24
        else:
            t_llm = 0.32

        per_step = 0.075 if self.scheduler == SchedulerType.DDPM else 0.073
        t_diff = per_step * decoded.steps
        t_metric = 0.08
        return round(t_llm + t_diff + t_metric, 3)

    def _compute_position_distance(self, position: List[float]) -> float:
        """Нормированное расстояние от предыдущей точки для оценки перцептивной гладкости LPIPS."""
        if self._prev_position is None or len(self._prev_position) != len(position):
            return 0.25
        sq_sum = 0.0
        for idx, (a, b) in enumerate(zip(self._prev_position, position)):
            scale = 12.0 if idx == 0 else (40.0 if idx == 1 else 1.0)
            sq_sum += ((a - b) / scale) ** 2
        return min(1.0, math.sqrt(sq_sum / max(1, len(position))))

    def evaluate_generation(
        self,
        decoded: DecodedPrompt,
        position: List[float],
        core_prompt: str,
        is_baseline: bool = False,
    ) -> AdapterOutput:
        num_slots = max(0, (len(position) - 2) // 2)
        weights = position[2 : 2 + num_slots]
        selectors = position[2 + num_slots : 2 + 2 * num_slots]

        opt_g, opt_w, opt_z = self._derive_prompt_optimum(core_prompt, num_slots)

        # 1. Влияние Guidance Scale (g) на семантику и визуальную целостность
        g_err = abs(decoded.guidance_scale - opt_g) / 6.0
        g_clip = math.exp(-0.7 * (g_err ** 2))
        g_vis = math.exp(-1.4 * (g_err ** 2))

        # 2. Влияние планировщика (DDPM vs DDIM) и числа шагов денойзинга (s)
        if self.scheduler == SchedulerType.DDIM:
            s_quality = 1.0 - 0.35 * math.exp(-decoded.steps / 6.5)
            noise_std = 0.005  # Детерминированный отклик при eta = 0
        else:
            s_quality = 1.0 - 0.60 * math.exp(-decoded.steps / 18.0)
            noise_std = 0.055  # Стохастический шум марковского процесса DDPM

        # 3. Расчет базовых метрик CLIPScore и ImageReward/Aesthetic
        if is_baseline:
            raw_clip = 0.58 * g_clip * (0.85 + 0.15 * s_quality)
            raw_vis = 0.52 * g_vis * s_quality
        else:
            if num_slots > 0:
                w_matches = [
                    math.exp(-2.2 * ((w - ow) ** 2))
                    for w, ow in zip(weights, opt_w)
                ]
                z_matches = [
                    math.exp(-3.0 * ((z - oz) ** 2))
                    for z, oz in zip(selectors, opt_z)
                ]
                avg_w = sum(w_matches) / num_slots
                avg_z = sum(z_matches) / num_slots
            else:
                avg_w, avg_z = 0.8, 0.8

            # Потеря качества при обрезке промпта за пределом контекстного окна CLIP
            overflow_ratio = max(
                0.0, (decoded.token_count - self.max_tokens) / float(self.max_tokens)
            )
            overflow_factor = max(0.65, 1.0 - 0.45 * overflow_ratio)

            # Учет управляемости промпта для каждой стратегии LLM
            if decoded.strategy == PromptStrategyType.NEUROSYMBOLIC:
                eff_w = avg_w
                strategy_factor = 1.0
            elif decoded.strategy == PromptStrategyType.IN_CONTEXT:
                eff_w = 0.55 + 0.30 * avg_w
                strategy_factor = 0.94
            else:
                eff_w = 0.42 + 0.20 * avg_w
                strategy_factor = 0.86

            raw_clip = (
                (0.47 + 0.45 * (0.45 * g_clip + 0.55 * avg_z))
                * decoded.core_retention_ratio
                * overflow_factor
                * (0.88 + 0.12 * s_quality)
                * strategy_factor
            )
            raw_vis = (
                (0.37 + 0.55 * (0.35 * g_vis + 0.35 * eff_w + 0.30 * avg_z))
                * s_quality
                * overflow_factor
                * strategy_factor
            )

        # 4. Наложение стохастичности планировщика
        clip_score = max(0.0, min(1.0, raw_clip + self._rng.gauss(0.0, noise_std)))
        vis_score = max(0.0, min(1.0, raw_vis + self._rng.gauss(0.0, noise_std)))

        # 5. Перцептивное расстояние LPIPS
        pos_dist = self._compute_position_distance(position)
        if self.scheduler == SchedulerType.DDPM:
            raw_lpips = 0.28 + 0.08 * (1.0 - s_quality) + 0.15 * pos_dist + self._rng.uniform(-0.02, 0.02)
        else:
            raw_lpips = 0.015 + 0.045 * (1.0 - s_quality) + 0.06 * pos_dist
        lpips_score = max(0.0, min(1.0, raw_lpips))

        if not is_baseline:
            self._prev_position = list(position)

        return AdapterOutput(
            clip_score=round(clip_score, 4),
            vis_score=round(vis_score, 4),
            lpips_score=round(lpips_score, 4),
            eval_time_sec=self._estimate_eval_time(decoded, is_baseline),
            image_url=None,
        )


class ExternalAPIDiffusionAdapter(BaseDiffusionAdapter):
    """
    Адаптер для генерации изображений через HuggingFace Inference API без локальной GPU.
    Требует переменную окружения HF_API_TOKEN.
    """

    def __init__(
        self,
        scheduler: SchedulerType = SchedulerType.DDIM,
        max_tokens: int = 77,
        seed: Optional[int] = 42,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        api_token: Optional[str] = None,
        timeout_sec: int = 45,
    ) -> None:
        super().__init__(scheduler=scheduler, max_tokens=max_tokens, seed=seed)
        self.model_id = model_id
        self.api_token = api_token or os.environ.get("HF_API_TOKEN")
        self.timeout_sec = timeout_sec
        self._surrogate_scorer = SurrogateDiffusionAdapter(
            scheduler=scheduler, max_tokens=max_tokens, seed=seed
        )

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed)
        self._surrogate_scorer.reset(seed)

    def evaluate_generation(
        self,
        decoded: DecodedPrompt,
        position: List[float],
        core_prompt: str,
        is_baseline: bool = False,
    ) -> AdapterOutput:
        if not self.api_token:
            raise RuntimeError(
                "Для режима backend_mode='api' необходимо задать переменную окружения HF_API_TOKEN."
            )

        url = f"https://router.huggingface.co/hf-inference/models/{self.model_id}"
        payload = json.dumps(
            {
                "inputs": decoded.prompt,
                "parameters": {
                    "guidance_scale": decoded.guidance_scale,
                    "num_inference_steps": decoded.steps,
                },
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as response:
                image_bytes = response.read()
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ошибка обращения к HuggingFace Inference API: {exc}") from exc

        elapsed = round(time.perf_counter() - started, 3)
        b64_img = base64.b64encode(image_bytes).decode("ascii")
        image_url = f"data:image/jpeg;base64,{b64_img}"

        # На Render (без PyTorch) метрики вычисляются через легкий суррогатный скорер
        proxy = self._surrogate_scorer.evaluate_generation(
            decoded=decoded,
            position=position,
            core_prompt=core_prompt,
            is_baseline=is_baseline,
        )
        return AdapterOutput(
            clip_score=proxy.clip_score,
            vis_score=proxy.vis_score,
            lpips_score=proxy.lpips_score,
            eval_time_sec=elapsed,
            image_url=image_url,
        )


class ColabLocalDiffusionAdapter(BaseDiffusionAdapter):
    """
    Реальный GPU-адаптер для Google Colab (Stable Diffusion + CLIPScore + DDPM/DDIM).
    Тяжелые библиотеки (torch, diffusers, transformers) импортируются лениво при первом вызове.
    """

    def __init__(
        self,
        scheduler: SchedulerType = SchedulerType.DDIM,
        max_tokens: int = 77,
        seed: Optional[int] = 42,
        sd_model_id: str = "runwayml/stable-diffusion-v1-5",
        clip_model_id: str = "openai/clip-vit-base-patch32",
        encode_images_base64: bool = True,
    ) -> None:
        super().__init__(scheduler=scheduler, max_tokens=max_tokens, seed=seed)
        self.sd_model_id = sd_model_id
        self.clip_model_id = clip_model_id
        self.encode_images_base64 = encode_images_base64
        self._pipe = None
        self._clip_model = None
        self._clip_processor = None
        self._device: Optional[str] = None
        self._prev_image_tensor = None
        self._call_counter: int = 0

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed)
        self._prev_image_tensor = None
        self._call_counter = 0

    def _load_models(self) -> None:
        if self._pipe is not None:
            return
        try:
            import torch
            from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Режим backend_mode='colab_local' требует установки torch, diffusers, "
                "transformers и pillow (запускайте этот режим в Google Colab с GPU)."
            ) from exc

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self._device == "cuda" else torch.float32

        pipe = StableDiffusionPipeline.from_pretrained(
            self.sd_model_id, torch_dtype=dtype, safety_checker=None
        )
        if self.scheduler == SchedulerType.DDIM:
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        else:
            pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

        self._pipe = pipe.to(self._device)
        self._clip_model = CLIPModel.from_pretrained(self.clip_model_id).to(self._device)
        self._clip_processor = CLIPProcessor.from_pretrained(self.clip_model_id)

    def evaluate_generation(
        self,
        decoded: DecodedPrompt,
        position: List[float],
        core_prompt: str,
        is_baseline: bool = False,
    ) -> AdapterOutput:
        import io
        import torch
        import torch.nn.functional as F

        self._load_models()
        started = time.perf_counter()
        self._call_counter += 1

        # Для DDIM фиксируем seed (строгая детерминированность), для DDPM — варьируем шаг
        effective_seed = (
            (self.seed or 42)
            if self.scheduler == SchedulerType.DDIM
            else (self.seed or 42) + self._call_counter
        )
        generator = torch.Generator(device=self._device).manual_seed(effective_seed)

        pipe_kwargs = {
            "prompt": decoded.prompt,
            "guidance_scale": decoded.guidance_scale,
            "num_inference_steps": decoded.steps,
            "generator": generator,
        }
        if self.scheduler == SchedulerType.DDIM:
            pipe_kwargs["eta"] = 0.0

        image = self._pipe(**pipe_kwargs).images[0]

        # 1. Реальный CLIPScore (косинусное сходство с исходным запросом T_0)
        inputs_core = self._clip_processor(
            text=[core_prompt, "high quality detailed aesthetic artwork, sharp focus", "blurry ugly deformed low quality artifact"],
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self._device)

        with torch.no_grad():
            outputs = self._clip_model(**inputs_core)
            img_emb = F.normalize(outputs.image_embeds, dim=-1)
            txt_embs = F.normalize(outputs.text_embeds, dim=-1)

            # Косинусное сходство с ядром запроса (типичный диапазон CLIP 0.18..0.38 -> нормируем в [0; 1])
            cos_core = float((img_emb @ txt_embs[0].unsqueeze(-1)).item())
            clip_score = max(0.0, min(1.0, (cos_core - 0.12) / 0.26))

            # Перцептивный контраст позитивного и негативного эстетического якоря
            cos_pos = float((img_emb @ txt_embs[1].unsqueeze(-1)).item())
            cos_neg = float((img_emb @ txt_embs[2].unsqueeze(-1)).item())
            vis_score = max(0.0, min(1.0, 0.5 + (cos_pos - cos_neg) * 3.5))

            # Перцептивное расстояние между текущим и прошлым эмбеддингом изображения
            if self._prev_image_tensor is not None:
                cos_prev = float((img_emb @ self._prev_image_tensor.T).item())
                lpips_score = max(0.0, min(1.0, 1.0 - cos_prev))
            else:
                lpips_score = 0.0
            if not is_baseline:
                self._prev_image_tensor = img_emb.detach()

        elapsed = round(time.perf_counter() - started, 3)

        image_url: Optional[str] = None
        if self.encode_images_base64:
            thumb = image.copy()
            thumb.thumbnail((256, 256))
            buf = io.BytesIO()
            thumb.save(buf, format="JPEG", quality=82)
            image_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

        return AdapterOutput(
            clip_score=round(clip_score, 4),
            vis_score=round(vis_score, 4),
            lpips_score=round(lpips_score, 4),
            eval_time_sec=elapsed,
            image_url=image_url,
        )


def create_diffusion_adapter(config: ExperimentConfig) -> BaseDiffusionAdapter:
    """Фабричный метод создания адаптера диффузии по конфигурации эксперимента."""
    if config.backend_mode == BackendMode.SURROGATE:
        return SurrogateDiffusionAdapter(
            scheduler=config.scheduler,
            max_tokens=config.max_tokens,
            seed=config.seed,
        )
    if config.backend_mode == BackendMode.API:
        return ExternalAPIDiffusionAdapter(
            scheduler=config.scheduler,
            max_tokens=config.max_tokens,
            seed=config.seed,
        )
    if config.backend_mode == BackendMode.COLAB_LOCAL:
        return ColabLocalDiffusionAdapter(
            scheduler=config.scheduler,
            max_tokens=config.max_tokens,
            seed=config.seed,
        )
    raise ValueError(f"Неподдерживаемый режим backend_mode: {config.backend_mode}")
