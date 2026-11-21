from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.error
import urllib.request

DEFAULT_CF_IMAGE_MODEL = "@cf/leonardo/lucid-origin"
_ENV_PATH = Path(__file__).resolve().parent / ".env"
_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ConvergeLab/1.3"
_WEIGHTED_SLOT_RE = re.compile(r"\(([^()]+):(\d+(?:\.\d+)?)\)")

# Предпочтительные качественные SD 1.5 модели на AI Horde (резервный провайдер)
_PREFERRED_HORDE_MODELS: Tuple[str, ...] = (
    "ICBINP - I Can't Believe It's Not Photography",
    "Deliberate",
    "DreamShaper",
    "Realistic Vision",
    "epiCRealism",
    "AbsoluteReality",
    "CyberRealistic",
    "stable_diffusion",
)


def _load_local_env() -> None:
    """Загружает переменные из backend/.env без перезаписи системных переменных (Render)."""
    if not _ENV_PATH.is_file():
        return
    try:
        for raw_line in _ENV_PATH.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        return


def get_hf_api_token() -> Optional[str]:
    """Сохранено для обратной совместимости."""
    _load_local_env()
    token = os.environ.get("HF_API_TOKEN", "").strip()
    return token or None


def get_cloudflare_credentials() -> Optional[Tuple[str, str]]:
    """Возвращает (account_id, api_token) для Cloudflare Workers AI, если оба заданы."""
    _load_local_env()
    account_id = os.environ.get("CF_ACCOUNT_ID", "").strip()
    api_token = os.environ.get("CF_API_TOKEN", "").strip()
    if account_id and api_token:
        return account_id, api_token
    return None


def is_hf_token_configured() -> bool:
    """Возвращает True (генерация работает через Cloudflare Workers AI или резервный AI Horde)."""
    return True


def sanitize_prompt_for_flux(prompt: str) -> str:
    """
    Убирает синтаксис весов вида '(phrase:1.14)' перед отправкой в Diffusers SDXL / FLUX,
    чтобы модель не интерпретировала скобки и числа как визуальный шум.
    """
    cleaned = _WEIGHTED_SLOT_RE.sub(lambda match: match.group(1).strip(), prompt)
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
    cleaned = re.sub(r"(?:,\s*){2,}", ", ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip(" ,")


def _parse_float(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Параметр '{name}' должен быть числом.")
    try:
        num = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Параметр '{name}' должен быть числом.") from exc
    return round(max(low, min(high, num)), 2)


def _parse_int(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Параметр '{name}' должен быть целым числом.")
    try:
        num = int(round(float(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Параметр '{name}' должен быть целым числом.") from exc
    return max(low, min(high, num))


@dataclass(frozen=True)
class SingleRenderOutput:
    image_url: str
    render_time_sec: float
    provider_used: str
    model_used: str


def _sniff_base64_mime(b64_str: str) -> str:
    if b64_str.startswith("/9j/"):
        return "image/jpeg"
    if b64_str.startswith("iVBOR"):
        return "image/png"
    if b64_str.startswith("UklGR"):
        return "image/webp"
    return "image/png"


def _extract_cf_error(raw_body: str) -> str:
    if not raw_body:
        return ""
    try:
        parsed = json.loads(raw_body)
        if isinstance(parsed, dict):
            errors = parsed.get("errors")
            if isinstance(errors, list) and errors:
                first = errors[0]
                if isinstance(first, dict) and isinstance(first.get("message"), str):
                    return first["message"]
            if isinstance(parsed.get("error"), str):
                return parsed["error"]
    except json.JSONDecodeError:
        pass
    return raw_body[:220]


def _call_cloudflare_once(
    url: str,
    api_token: str,
    payload: Dict[str, Any],
    model_id: str,
    timeout_sec: int,
) -> SingleRenderOutput:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
            "User-Agent": _USER_AGENT,
        },
        method="POST",
    )

    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        content_type = resp.headers.get("Content-Type", "")
        raw_bytes = resp.read()
    elapsed = round(time.perf_counter() - started, 3)

    mime = content_type.split(";")[0].strip().lower()
    if mime.startswith("image/"):
        if len(raw_bytes) < 500:
            raise RuntimeError("Cloudflare Workers AI вернул пустое изображение.")
        b64 = base64.b64encode(raw_bytes).decode("ascii")
        image_url = f"data:{mime};base64,{b64}"
    else:
        text = raw_bytes.decode("utf-8", errors="ignore")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Неожиданный ответ Cloudflare Workers AI ({mime}): {text[:180]}"
            ) from exc

        result = parsed.get("result") if isinstance(parsed, dict) else None
        img_b64 = result.get("image") if isinstance(result, dict) else None
        if not isinstance(img_b64, str) or not img_b64:
            detail = _extract_cf_error(text)
            raise RuntimeError(
                f"Cloudflare Workers AI не вернул изображение: {detail or 'неизвестный ответ'}"
            )
        detected_mime = _sniff_base64_mime(img_b64)
        image_url = f"data:{detected_mime};base64,{img_b64}"

    short_model = model_id.split("/")[-1]
    return SingleRenderOutput(
        image_url=image_url,
        render_time_sec=elapsed,
        provider_used="cloudflare_workers_ai",
        model_used=f"cf/{short_model}",
    )


def _generate_via_cloudflare(
    prompt: str,
    guidance_scale: float,
    steps: int,
    seed: int,
    account_id: str,
    api_token: str,
    model_id: str,
    timeout_sec: int = 60,
) -> SingleRenderOutput:
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model_id}"
    clean_prompt = sanitize_prompt_for_flux(prompt)[:1500]
    is_flux = "flux" in model_id.lower()

    if is_flux:
        capped_steps = max(4, min(8, steps))
        attempts: List[Dict[str, Any]] = [
            {"prompt": clean_prompt, "steps": capped_steps, "seed": seed},
            {"prompt": clean_prompt, "steps": capped_steps},
        ]
    else:
        capped_steps = max(10, min(20, steps))
        attempts = [
            {
                "prompt": clean_prompt,
                "width": 768,
                "height": 768,
                "guidance": guidance_scale,
                "num_steps": capped_steps,
                "seed": seed,
            },
            {
                "prompt": clean_prompt,
                "guidance": guidance_scale,
                "num_steps": capped_steps,
            },
        ]

    last_error: Optional[Exception] = None
    for idx, payload in enumerate(attempts):
        try:
            return _call_cloudflare_once(url, api_token, payload, model_id, timeout_sec)
        except urllib.error.HTTPError as exc:
            raw_err = exc.read().decode("utf-8", errors="ignore")
            detail = _extract_cf_error(raw_err)
            if exc.code == 400 and idx + 1 < len(attempts):
                last_error = RuntimeError(f"Cloudflare HTTP 400: {detail}")
                continue
            if exc.code in (401, 403):
                raise RuntimeError(
                    "Неверный CF_ACCOUNT_ID или CF_API_TOKEN (проверьте права Workers AI)."
                ) from exc
            if exc.code == 429:
                raise RuntimeError(
                    "Исчерпан дневной бесплатный лимит Cloudflare Workers AI (10 000 Neurons)."
                ) from exc
            raise RuntimeError(
                f"Ошибка Cloudflare Workers AI (HTTP {exc.code}): {detail or exc.reason}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Cloudflare Workers AI недоступен: {exc}") from exc

    assert last_error is not None
    raise last_error


def _render_pair_via_cloudflare(
    base_prompt: str,
    opt_prompt: str,
    base_g: float,
    base_s: int,
    opt_g: float,
    opt_s: int,
    seed: int,
    account_id: str,
    api_token: str,
    model_id: str,
) -> Tuple[SingleRenderOutput, SingleRenderOutput]:
    """Генерирует оба кадра параллельно через Cloudflare Workers AI на одной модели с общим seed."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_base = pool.submit(
            _generate_via_cloudflare,
            base_prompt,
            base_g,
            base_s,
            seed,
            account_id,
            api_token,
            model_id,
        )
        fut_opt = pool.submit(
            _generate_via_cloudflare,
            opt_prompt,
            opt_g,
            opt_s,
            seed,
            account_id,
            api_token,
            model_id,
        )
        return fut_base.result(), fut_opt.result()


def _horde_request_json(
    url: str,
    api_key: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout_sec: int = 20,
) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "apikey": api_key,
            "Client-Agent": "ConvergeLab:1.3:academic",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": _USER_AGENT,
        },
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _pick_fastest_horde_model(api_key: str) -> Optional[str]:
    """Выбирает самую быструю активную качественную модель AI Horde по минимальному ETA."""
    try:
        models_raw = _horde_request_json(
            "https://aihorde.net/api/v2/status/models?type=image",
            api_key=api_key,
            timeout_sec=8,
        )
    except Exception:
        return None

    if not isinstance(models_raw, list):
        return None

    preferred_set = set(_PREFERRED_HORDE_MODELS)
    preferred_candidates: List[Tuple[int, int, str]] = []
    general_candidates: List[Tuple[int, int, str]] = []
    banned_words = ("inpaint", "controlnet", "pix2pix", "nsfw", "hentai", "pony")

    for entry in models_raw:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        count = entry.get("count", 0)
        eta = entry.get("eta", 9999)
        if not isinstance(name, str) or not isinstance(count, int) or count <= 0:
            continue
        eta_val = int(eta) if isinstance(eta, (int, float)) else 9999
        lower_name = name.lower()
        if any(bad in lower_name for bad in banned_words):
            continue

        if name in preferred_set:
            preferred_candidates.append((eta_val, -count, name))
        elif count >= 2:
            general_candidates.append((eta_val, -count, name))

    if preferred_candidates:
        preferred_candidates.sort()
        return preferred_candidates[0][2]
    if general_candidates:
        general_candidates.sort()
        return general_candidates[0][2]
    return None


def _generate_via_ai_horde(
    prompt: str,
    guidance_scale: float,
    steps: int,
    seed: int,
    api_key: str,
    pinned_model: Optional[str] = None,
    timeout_sec: int = 110,
) -> SingleRenderOutput:
    base_url = "https://aihorde.net/api/v2"

    payload: Dict[str, Any] = {
        "prompt": prompt,
        "params": {
            "sampler_name": "k_euler",
            "cfg_scale": round(max(1.0, min(20.0, guidance_scale)), 2),
            "steps": max(15, min(25, steps)),
            "width": 512,
            "height": 512,
            "seed": str(seed),
            "n": 1,
        },
        "nsfw": False,
        "censor_nsfw": True,
        "trusted_workers": False,
        "slow_workers": True,
        "r2": True,
        "shared": False,
    }
    if pinned_model:
        payload["models"] = [pinned_model]

    started = time.perf_counter()
    try:
        submitted = _horde_request_json(
            f"{base_url}/generate/async",
            api_key=api_key,
            payload=payload,
            timeout_sec=20,
        )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")[:200]
        raise RuntimeError(f"AI Horde HTTP {exc.code}: {body or exc.reason}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"AI Horde недоступен: {exc}") from exc

    job_id = submitted.get("id") if isinstance(submitted, dict) else None
    if not isinstance(job_id, str) or not job_id:
        raise RuntimeError("AI Horde не вернул идентификатор задачи.")

    deadline = started + timeout_sec
    while time.perf_counter() < deadline:
        time.sleep(2.5)
        try:
            check = _horde_request_json(
                f"{base_url}/generate/check/{job_id}",
                api_key=api_key,
                timeout_sec=15,
            )
        except Exception:
            continue

        if not isinstance(check, dict):
            continue
        if check.get("faulted"):
            raise RuntimeError("Воркер AI Horde завершил генерацию с ошибкой.")
        if check.get("is_possible") is False:
            raise RuntimeError("Нет свободных воркеров AI Horde для выбранной модели.")
        if check.get("done"):
            break
    else:
        raise RuntimeError(
            "Очередь AI Horde перегружена. Для быстрой генерации (~10 сек) добавьте бесплатные "
            "CF_ACCOUNT_ID и CF_API_TOKEN от Cloudflare Workers AI в backend/.env."
        )

    status = _horde_request_json(
        f"{base_url}/generate/status/{job_id}",
        api_key=api_key,
        timeout_sec=20,
    )
    generations = status.get("generations") if isinstance(status, dict) else None
    if not isinstance(generations, list) or not generations:
        raise RuntimeError("AI Horde вернул пустой список изображений.")

    first = generations[0]
    img_field = first.get("img", "")
    worker_model = first.get("model") or pinned_model or "stable_diffusion"
    if not isinstance(img_field, str) or not img_field:
        raise RuntimeError("AI Horde не вернул данные изображения.")

    if img_field.startswith("http"):
        img_req = urllib.request.Request(
            img_field,
            headers={"User-Agent": _USER_AGENT},
            method="GET",
        )
        with urllib.request.urlopen(img_req, timeout=20) as img_resp:
            content_type = img_resp.headers.get("Content-Type", "image/webp")
            image_bytes = img_resp.read()
        mime = content_type.split(";")[0].strip().lower()
        if not mime.startswith("image/"):
            mime = "image/webp"
        b64 = base64.b64encode(image_bytes).decode("ascii")
        image_url = f"data:{mime};base64,{b64}"
    else:
        image_url = f"data:image/webp;base64,{img_field}"

    elapsed = round(time.perf_counter() - started, 3)
    return SingleRenderOutput(
        image_url=image_url,
        render_time_sec=elapsed,
        provider_used="aihorde",
        model_used=f"aihorde/{worker_model}",
    )


def _render_pair_via_ai_horde(
    base_prompt: str,
    opt_prompt: str,
    base_g: float,
    base_s: int,
    opt_g: float,
    opt_s: int,
    seed: int,
    timeout_sec: int = 110,
) -> Tuple[SingleRenderOutput, SingleRenderOutput]:
    """Рендерит оба кадра параллельно на самой быстрой активной модели AI Horde."""
    _load_local_env()
    api_key = os.environ.get("AI_HORDE_API_KEY", "").strip() or "0000000000"
    pinned_model = _pick_fastest_horde_model(api_key)

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_base = pool.submit(
            _generate_via_ai_horde,
            base_prompt,
            base_g,
            base_s,
            seed,
            api_key,
            pinned_model,
            timeout_sec,
        )
        fut_opt = pool.submit(
            _generate_via_ai_horde,
            opt_prompt,
            opt_g,
            opt_s,
            seed,
            api_key,
            pinned_model,
            timeout_sec,
        )
        return fut_base.result(), fut_opt.result()


def render_before_after_pair(
    baseline_prompt: Any,
    optimized_prompt: Any,
    baseline_guidance_scale: Any = 7.5,
    baseline_steps: Any = 25,
    optimized_guidance_scale: Any = 8.0,
    optimized_steps: Any = 25,
    seed: Any = None,
    model_id: Optional[str] = None,
    timeout_sec: int = 110,
) -> Dict[str, Any]:
    """
    Атомарно генерирует пару изображений «До / После» на ОДНОМ провайдере и ОДНОЙ модели
    с общим seed:
    - Cloudflare Workers AI (если заданы CF_ACCOUNT_ID и CF_API_TOKEN — быстро, ~8–12 сек)
    - AI Horde (автоматический бесплатный резерв без ключей)
    """
    if not isinstance(baseline_prompt, str) or not baseline_prompt.strip():
        raise ValueError("baseline_prompt должен быть непустой строкой.")
    if not isinstance(optimized_prompt, str) or not optimized_prompt.strip():
        raise ValueError("optimized_prompt должен быть непустой строкой.")

    base_prompt = baseline_prompt.strip()
    opt_prompt = optimized_prompt.strip()

    base_g = _parse_float(baseline_guidance_scale, "baseline_guidance_scale", 1.0, 20.0)
    base_s = _parse_int(baseline_steps, "baseline_steps", 4, 50)
    opt_g = _parse_float(optimized_guidance_scale, "optimized_guidance_scale", 1.0, 20.0)
    opt_s = _parse_int(optimized_steps, "optimized_steps", 4, 50)

    if seed is None or seed == "":
        effective_seed = random.randint(1, 2_147_483_646)
    else:
        effective_seed = _parse_int(seed, "seed", 0, 2_147_483_646)

    _load_local_env()
    provider_mode = (os.environ.get("IMAGE_PROVIDER") or "auto").strip().lower()
    raw_model = model_id if isinstance(model_id, str) and model_id.strip() else None
    cf_model = (
        raw_model or os.environ.get("CF_IMAGE_MODEL") or DEFAULT_CF_IMAGE_MODEL
    ).strip()

    cf_creds = get_cloudflare_credentials()
    wall_start = time.perf_counter()

    if provider_mode == "cloudflare":
        if not cf_creds:
            raise RuntimeError(
                "Укажите CF_ACCOUNT_ID и CF_API_TOKEN в backend/.env для режима cloudflare."
            )
        base_out, opt_out = _render_pair_via_cloudflare(
            base_prompt,
            opt_prompt,
            base_g,
            base_s,
            opt_g,
            opt_s,
            effective_seed,
            cf_creds[0],
            cf_creds[1],
            cf_model,
        )
    elif provider_mode == "aihorde":
        base_out, opt_out = _render_pair_via_ai_horde(
            base_prompt,
            opt_prompt,
            base_g,
            base_s,
            opt_g,
            opt_s,
            effective_seed,
            timeout_sec,
        )
    else:
        if cf_creds:
            base_out, opt_out = _render_pair_via_cloudflare(
                base_prompt,
                opt_prompt,
                base_g,
                base_s,
                opt_g,
                opt_s,
                effective_seed,
                cf_creds[0],
                cf_creds[1],
                cf_model,
            )
        else:
            base_out, opt_out = _render_pair_via_ai_horde(
                base_prompt,
                opt_prompt,
                base_g,
                base_s,
                opt_g,
                opt_s,
                effective_seed,
                timeout_sec,
            )

    return {
        "model_id": opt_out.model_used,
        "seed": effective_seed,
        "source": opt_out.provider_used,
        "total_wall_time_sec": round(time.perf_counter() - wall_start, 3),
        "baseline": {
            "prompt": base_prompt,
            "guidance_scale": base_g,
            "steps": base_s,
            "image_url": base_out.image_url,
            "render_time_sec": base_out.render_time_sec,
        },
        "optimized": {
            "prompt": opt_prompt,
            "guidance_scale": opt_g,
            "steps": opt_s,
            "image_url": opt_out.image_url,
            "render_time_sec": opt_out.render_time_sec,
        },
    }
