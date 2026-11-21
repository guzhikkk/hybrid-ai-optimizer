from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

from flask import Flask, Response, jsonify, request
from flask_cors import CORS

# Гарантируем корректные импорты при запуске как из backend/, так и из корня репозитория
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from experiment_runner import ExperimentFactory, ExperimentRunner
from image_renderer import is_hf_token_configured, render_before_after_pair
from models import (
    BackendMode,
    ExperimentConfig,
    OptimizerType,
    PromptStrategyType,
    SchedulerType,
)

# Ограничения публичного API для стабильной работы на бесплатном тарифе Render
MAX_API_ITERATIONS = 30
MAX_API_PARTICLES = 20
MAX_API_RUNS = 10


def _parse_json_body() -> Dict[str, Any]:
    """Безопасно извлекает JSON-словарь из тела POST-запроса."""
    if not request.data:
        return {}
    payload = request.get_json(silent=True)
    if payload is None:
        raise ValueError("Некорректный JSON в теле запроса.")
    if not isinstance(payload, dict):
        raise ValueError("Тело запроса должно быть JSON-объектом.")
    return payload


def _validate_api_limits(config: ExperimentConfig) -> None:
    """Проверяет лимиты нагрузки для защиты сервера от таймаута."""
    if config.max_iterations > MAX_API_ITERATIONS:
        raise ValueError(
            f"max_iterations не должно превышать {MAX_API_ITERATIONS} в API-режиме."
        )
    if config.num_particles > MAX_API_PARTICLES:
        raise ValueError(
            f"num_particles не должно превышать {MAX_API_PARTICLES} в API-режиме."
        )


def create_app() -> Flask:
    """Фабрика приложения Flask (используется сервером Gunicorn и тестами pytest)."""
    flask_app = Flask(__name__)
    CORS(flask_app)

    @flask_app.route("/", methods=["GET"])
    @flask_app.route("/api/health", methods=["GET"])
    def health() -> Tuple[Response, int]:
        return (
            jsonify(
                {
                    "status": "ok",
                    "service": "hybrid-ai-optimizer-backend",
                    "version": "1.1.0",
                    "image_rendering_configured": is_hf_token_configured(),
                }
            ),
            200,
        )

    @flask_app.route("/api/config/defaults", methods=["GET"])
    def get_defaults() -> Tuple[Response, int]:
        default_config = ExperimentConfig()
        presets = [
            {
                "preset_id": p.preset_id,
                "label": p.label,
                "optimizer": p.optimizer.value,
                "prompt_strategy": p.prompt_strategy.value,
                "scheduler": p.scheduler.value,
            }
            for p in ExperimentFactory.default_comparison_presets()
        ]
        return (
            jsonify(
                {
                    "defaults": default_config.to_dict(),
                    "options": {
                        "optimizers": [e.value for e in OptimizerType],
                        "prompt_strategies": [e.value for e in PromptStrategyType],
                        "schedulers": [e.value for e in SchedulerType],
                        "backend_modes": [e.value for e in BackendMode],
                    },
                    "comparison_presets": presets,
                }
            ),
            200,
        )

    @flask_app.route("/api/optimize", methods=["POST"])
    def run_optimize() -> Tuple[Response, int]:
        """Одиночный цикл оптимизации (для прикладной вкладки 'Оптимизация')."""
        try:
            payload = _parse_json_body()
            config = ExperimentConfig.from_dict(payload)
            _validate_api_limits(config)

            runner = ExperimentRunner()
            result = runner.run_experiment(config)
            return jsonify(result.to_dict()), 200
        except (ValueError, TypeError) as exc:
            return jsonify({"error": str(exc)}), 400
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 503
        except Exception as exc:
            return (
                jsonify({"error": "Внутренняя ошибка сервера", "details": str(exc)}),
                500,
            )

    @flask_app.route("/api/compare", methods=["POST"])
    def run_compare() -> Tuple[Response, int]:
        """Сравнительный анализ всех 4 пресетов (для исследовательской вкладки 'Сравнение')."""
        try:
            payload = _parse_json_body()
            raw_runs = payload.get("num_runs", 3)
            if isinstance(raw_runs, bool):
                raise ValueError("num_runs должно быть целым числом.")
            num_runs = int(raw_runs)
            if not (1 <= num_runs <= MAX_API_RUNS):
                raise ValueError(
                    f"num_runs должно лежать в диапазоне [1, {MAX_API_RUNS}]."
                )

            config = ExperimentConfig.from_dict(payload)
            _validate_api_limits(config)

            runner = ExperimentRunner()
            comparison = runner.run_comparison(config, num_runs=num_runs)
            return jsonify(comparison), 200
        except (ValueError, TypeError) as exc:
            return jsonify({"error": str(exc)}), 400
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 503
        except Exception as exc:
            return (
                jsonify({"error": "Внутренняя ошибка сервера", "details": str(exc)}),
                500,
            )

    @flask_app.route("/api/render-pair", methods=["POST"])
    def run_render_pair() -> Tuple[Response, int]:
        """Генерация пары изображений «До / После» с общим seed через HuggingFace API."""
        try:
            payload = _parse_json_body()
            rendered = render_before_after_pair(
                baseline_prompt=payload.get("baseline_prompt", ""),
                optimized_prompt=payload.get("optimized_prompt", ""),
                baseline_guidance_scale=payload.get("baseline_guidance_scale", 7.5),
                baseline_steps=payload.get("baseline_steps", 25),
                optimized_guidance_scale=payload.get("optimized_guidance_scale", 8.0),
                optimized_steps=payload.get("optimized_steps", 25),
                seed=payload.get("seed"),
                model_id=payload.get("model_id"),
            )
            return jsonify(rendered), 200
        except (ValueError, TypeError) as exc:
            return jsonify({"error": str(exc)}), 400
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 503
        except Exception as exc:
            return (
                jsonify({"error": "Внутренняя ошибка сервера", "details": str(exc)}),
                500,
            )

    return flask_app


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
