import { useCallback, useEffect, useRef, useState } from 'react';
import type { KeyboardEvent, PointerEvent } from 'react';
import {
  AlertCircle,
  Columns2,
  Download,
  Image as ImageIcon,
  Loader2,
  MoveHorizontal,
  RefreshCw,
  Sparkles,
} from 'lucide-react';
import { ApiError, renderComparisonPair } from '../services/apiClient';
import type { ExperimentResult, RenderPairResponse } from '../types/api';

type CompareViewMode = 'side-by-side' | 'slider';

interface VisualCompareCardProps {
  result: ExperimentResult | null;
  isOptimizing: boolean;
  imageRenderingConfigured: boolean | null;
}

function loadImageElement(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new window.Image();
    img.onload = () => resolve(img);
    img.onerror = () =>
      reject(new Error('Не удалось загрузить изображение для экспорта PNG.'));
    img.src = src;
  });
}

function drawRoundedRectPath(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
): void {
  const radius = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + w - radius, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
  ctx.lineTo(x + w, y + h - radius);
  ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
  ctx.lineTo(x + radius, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

function drawCoverImage(
  ctx: CanvasRenderingContext2D,
  img: HTMLImageElement,
  x: number,
  y: number,
  w: number,
  h: number,
  radius: number,
): void {
  const imgRatio = img.naturalWidth / Math.max(1, img.naturalHeight);
  const boxRatio = w / Math.max(1, h);

  let sx = 0;
  let sy = 0;
  let sw = img.naturalWidth;
  let sh = img.naturalHeight;

  if (imgRatio > boxRatio) {
    sw = img.naturalHeight * boxRatio;
    sx = (img.naturalWidth - sw) / 2;
  } else {
    sh = img.naturalWidth / boxRatio;
    sy = (img.naturalHeight - sh) / 2;
  }

  ctx.save();
  drawRoundedRectPath(ctx, x, y, w, h, radius);
  ctx.clip();
  ctx.drawImage(img, sx, sy, sw, sh, x, y, w, h);
  ctx.restore();
}

export function VisualCompareCard({
  result,
  isOptimizing,
  imageRenderingConfigured,
}: VisualCompareCardProps) {
  const [viewMode, setViewMode] = useState<CompareViewMode>('side-by-side');
  const [rendered, setRendered] = useState<RenderPairResponse | null>(null);
  const [isRendering, setIsRendering] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sliderPos, setSliderPos] = useState(50);

  const sliderStageRef = useRef<HTMLDivElement | null>(null);
  const isDraggingRef = useRef(false);
  const requestIdRef = useRef(0);

  const bestEval =
    result && result.history.length > 0
      ? result.history[result.history.length - 1].global_best_evaluation
      : null;

  const baselineSteps = result?.config.scheduler === 'ddpm' ? 50 : 25;
  const baselineGuidance = 7.5;

  // Сбрасываем старую пару картинок, когда выполнен новый запуск оптимизации
  useEffect(() => {
    requestIdRef.current += 1;
    setRendered(null);
    setError(null);
    setIsRendering(false);
    setSliderPos(50);
  }, [result?.experiment_id]);

  const handleRender = async (customSeed?: number) => {
    if (!result || !bestEval || isRendering) return;

    const currentReqId = requestIdRef.current + 1;
    requestIdRef.current = currentReqId;

    setIsRendering(true);
    setError(null);

    const seedToUse =
      typeof customSeed === 'number'
        ? customSeed
        : (result.config.seed ?? Math.floor(Math.random() * 999_999) + 1);

    try {
      const response = await renderComparisonPair({
        baseline_prompt: result.config.core_prompt,
        optimized_prompt: bestEval.prompt,
        baseline_guidance_scale: baselineGuidance,
        baseline_steps: baselineSteps,
        optimized_guidance_scale: bestEval.guidance_scale,
        optimized_steps: bestEval.steps,
        seed: seedToUse,
      });

      if (requestIdRef.current !== currentReqId) return;
      setRendered(response);
    } catch (err: unknown) {
      if (requestIdRef.current !== currentReqId) return;
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError('Не удалось сгенерировать визуальное сравнение.');
      }
    } finally {
      if (requestIdRef.current === currentReqId) {
        setIsRendering(false);
      }
    }
  };

  const handleNewSeed = () => {
    const freshSeed = Math.floor(Math.random() * 999_999) + 1;
    void handleRender(freshSeed);
  };

  const handleExportCollage = async () => {
    if (!rendered || !result || !bestEval || isExporting) return;

    setIsExporting(true);
    setError(null);
    try {
      const [baseImg, optImg] = await Promise.all([
        loadImageElement(rendered.baseline.image_url),
        loadImageElement(rendered.optimized.image_url),
      ]);

      const canvas = document.createElement('canvas');
      canvas.width = 1280;
      canvas.height = 760;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        throw new Error('Canvas 2D недоступен в этом браузере.');
      }

      // Фон (Google Dark)
      ctx.fillStyle = '#131314';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Заголовок
      ctx.fillStyle = '#E3E3E3';
      ctx.font = '700 24px Inter, sans-serif';
      ctx.fillText('ConvergeLab — Визуальное сравнение «До / После»', 48, 52);

      ctx.fillStyle = '#9AA0A6';
      ctx.font = '500 14px "JetBrains Mono", monospace';
      const gainStr = `${result.macro_metrics.relative_gain_pct >= 0 ? '+' : ''}${result.macro_metrics.relative_gain_pct.toFixed(1)}%`;
      ctx.fillText(
        `Seed: ${rendered.seed}  |  Model: ${rendered.model_id}  |  ΔF: ${gainStr}`,
        48,
        80,
      );

      // Карточка слева (До)
      drawRoundedRectPath(ctx, 48, 108, 576, 604, 20);
      ctx.fillStyle = '#1E1F20';
      ctx.fill();
      ctx.strokeStyle = '#2E3033';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      ctx.fillStyle = '#E3E3E3';
      ctx.font = '600 16px Inter, sans-serif';
      ctx.fillText('До · Бейзлайн (сырой запрос T₀)', 72, 144);

      ctx.fillStyle = '#9AA0A6';
      ctx.font = '500 13px "JetBrains Mono", monospace';
      ctx.fillText(
        `F₀ = ${result.macro_metrics.baseline_fitness.toFixed(4)}  |  g = ${rendered.baseline.guidance_scale.toFixed(1)}  |  s = ${rendered.baseline.steps}`,
        72,
        168,
      );
      drawCoverImage(ctx, baseImg, 72, 188, 528, 500, 14);

      // Карточка справа (После)
      drawRoundedRectPath(ctx, 656, 108, 576, 604, 20);
      ctx.fillStyle = '#1E1F20';
      ctx.fill();
      ctx.strokeStyle = '#8AB4F8';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      ctx.fillStyle = '#8AB4F8';
      ctx.font = '600 16px Inter, sans-serif';
      ctx.fillText('После · ConvergeLab (оптимизировано)', 680, 144);

      ctx.fillStyle = '#81C995';
      ctx.font = '600 13px "JetBrains Mono", monospace';
      ctx.fillText(
        `F_best = ${bestEval.fitness.toFixed(4)} (${gainStr})  |  g = ${rendered.optimized.guidance_scale.toFixed(2)}  |  s = ${rendered.optimized.steps}`,
        680,
        168,
      );
      drawCoverImage(ctx, optImg, 680, 188, 528, 500, 14);

      const pngUrl = canvas.toDataURL('image/png');
      const link = document.createElement('a');
      link.href = pngUrl;
      link.download = `convergelab-compare-seed-${rendered.seed}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (err: unknown) {
      setError(
        err instanceof Error
          ? err.message
          : 'Не удалось сохранить PNG-коллаж.',
      );
    } finally {
      setIsExporting(false);
    }
  };

  const updateSliderFromClientX = useCallback((clientX: number) => {
    const el = sliderStageRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    if (rect.width <= 0) return;
    const pct = ((clientX - rect.left) / rect.width) * 100;
    setSliderPos(Math.max(4, Math.min(96, Math.round(pct * 10) / 10)));
  }, []);

  const handlePointerDown = (e: PointerEvent<HTMLDivElement>) => {
    isDraggingRef.current = true;
    e.currentTarget.setPointerCapture(e.pointerId);
    updateSliderFromClientX(e.clientX);
  };

  const handlePointerMove = (e: PointerEvent<HTMLDivElement>) => {
    if (!isDraggingRef.current) return;
    updateSliderFromClientX(e.clientX);
  };

  const handlePointerUp = (e: PointerEvent<HTMLDivElement>) => {
    isDraggingRef.current = false;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
  };

  const handleSliderKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'ArrowLeft') {
      e.preventDefault();
      setSliderPos((prev) => Math.max(4, prev - 5));
    } else if (e.key === 'ArrowRight') {
      e.preventDefault();
      setSliderPos((prev) => Math.min(96, prev + 5));
    }
  };

  const gainPct = result?.macro_metrics.relative_gain_pct ?? 0;

  return (
    <section className="card" aria-label="Визуальное сравнение До и После">
      <div className="card-header">
        <div>
          <h2 className="card-title">Визуальное сравнение «До / После»</h2>
          <p className="card-subtitle">
            Параллельная генерация бейзлайна и найденной конфигурации с единым seed
          </p>
        </div>

        <div className="vc-actions">
          {rendered && (
            <div className="axis-toggle" role="group" aria-label="Режим просмотра">
              <button
                type="button"
                className={`axis-toggle__btn ${
                  viewMode === 'side-by-side' ? 'axis-toggle__btn--active' : ''
                }`}
                onClick={() => setViewMode('side-by-side')}
              >
                <Columns2 size={14} style={{ marginRight: 4, verticalAlign: -2 }} />
                Рядом
              </button>
              <button
                type="button"
                className={`axis-toggle__btn ${
                  viewMode === 'slider' ? 'axis-toggle__btn--active' : ''
                }`}
                onClick={() => setViewMode('slider')}
              >
                <MoveHorizontal size={14} style={{ marginRight: 4, verticalAlign: -2 }} />
                Шторка
              </button>
            </div>
          )}

          {rendered && (
            <>
              <button
                type="button"
                className="btn-tonal"
                onClick={handleNewSeed}
                disabled={isRendering || isOptimizing}
              >
                <RefreshCw size={16} />
                <span>Новый seed</span>
              </button>

              <button
                type="button"
                className="btn-tonal"
                onClick={() => void handleExportCollage()}
                disabled={isRendering || isExporting}
              >
                <Download size={16} />
                <span>{isExporting ? 'Сохранение...' : 'Скачать PNG'}</span>
              </button>
            </>
          )}

          <button
            type="button"
            className="btn-primary"
            onClick={() => void handleRender()}
            disabled={!result || !bestEval || isRendering || isOptimizing}
          >
            {isRendering ? (
              <>
                <Loader2 size={16} className="spin-icon" />
                <span>Генерация...</span>
              </>
            ) : (
              <>
                <Sparkles size={16} />
                <span>
                  {rendered ? 'Перегенерировать' : 'Сгенерировать сравнение'}
                </span>
              </>
            )}
          </button>
        </div>
      </div>

      {imageRenderingConfigured === false && !rendered && !error && (
        <div className="vc-notice vc-notice--warning" role="status">
          <AlertCircle size={18} className="vc-notice__icon" />
          <p className="vc-notice__text">
            Токен <strong className="mono">HF_API_TOKEN</strong> пока не найден на
            сервере. Добавь его в <span className="mono">backend/.env</span> и
            перезапусти Flask, чтобы включить живую генерацию изображений.
          </p>
        </div>
      )}

      {error && (
        <div className="vc-notice vc-notice--error" role="alert">
          <AlertCircle size={18} className="vc-notice__icon" />
          <p className="vc-notice__text">{error}</p>
        </div>
      )}

      {!result || !bestEval ? (
        <div className="vc-empty">
          <span className="vc-empty__icon">
            <ImageIcon size={24} />
          </span>
          <h3 className="vc-empty__title">Готово к визуальному сравнению</h3>
          <p className="vc-empty__desc">
            Сначала запусти оптимизацию выше — после этого здесь можно в один клик
            сравнить картинку по сырому запросу и по оптимизированному промпту.
          </p>
        </div>
      ) : isRendering ? (
        <div className="vc-pair-grid">
          <div className="vc-frame">
            <div className="vc-frame__head">
              <span className="vc-frame__title">До · Бейзлайн (T₀)</span>
              <span className="pill-badge mono" style={{ background: 'var(--bg-surface-high)', color: 'var(--text-secondary)' }}>
                F₀ = {result.macro_metrics.baseline_fitness.toFixed(4)}
              </span>
            </div>
            <div className="vc-frame__media">
              <div className="vc-skeleton-box">
                <Loader2 size={24} className="spin-icon" />
                <span>Генерируем базовый кадр (обычно 6–15 сек)...</span>
              </div>
            </div>
          </div>

          <div className="vc-frame vc-frame--winner">
            <div className="vc-frame__head">
              <span className="vc-frame__title">После · ConvergeLab</span>
              <span className="pill-badge mono" style={{ background: 'var(--status-success-soft)', color: 'var(--status-success)' }}>
                F_best = {bestEval.fitness.toFixed(4)}
              </span>
            </div>
            <div className="vc-frame__media">
              <div className="vc-skeleton-box">
                <Loader2 size={24} className="spin-icon" />
                <span>Генерируем оптимизированный кадр с тем же seed...</span>
              </div>
            </div>
          </div>
        </div>
      ) : !rendered ? (
        <div className="vc-empty">
          <span className="vc-empty__icon">
            <ImageIcon size={24} />
          </span>
          <h3 className="vc-empty__title">Сравнить финальные кадры «До / После»</h3>
          <p className="vc-empty__desc">
            Оба изображения сгенерируются параллельно из одного и того же начального
            шума (seed), чтобы показать чистый эффект оптимизации промпта и параметров.
          </p>

          <div className="vc-preview-prompts">
            <div className="vc-prompt-box">
              <div className="vc-prompt-box__head mono">
                <span>До · Бейзлайн</span>
                <span>
                  g={baselineGuidance.toFixed(1)} · s={baselineSteps}
                </span>
              </div>
              <p className="vc-prompt-box__text mono">{result.config.core_prompt}</p>
            </div>

            <div className="vc-prompt-box">
              <div className="vc-prompt-box__head mono">
                <span style={{ color: 'var(--accent-blue)' }}>После · ConvergeLab</span>
                <span style={{ color: 'var(--status-success)' }}>
                  g={bestEval.guidance_scale.toFixed(2)} · s={bestEval.steps}
                </span>
              </div>
              <p className="vc-prompt-box__text mono">{bestEval.prompt}</p>
            </div>
          </div>
        </div>
      ) : viewMode === 'side-by-side' ? (
        <div className="vc-pair-grid">
          <article className="vc-frame">
            <div className="vc-frame__head">
              <span className="vc-frame__title">До · Бейзлайн (сырой запрос T₀)</span>
              <span
                className="pill-badge mono"
                style={{
                  backgroundColor: 'var(--bg-surface-high)',
                  color: 'var(--text-secondary)',
                }}
              >
                F₀ = {result.macro_metrics.baseline_fitness.toFixed(4)}
              </span>
            </div>

            <div className="vc-frame__media">
              <img
                className="vc-frame__img"
                src={rendered.baseline.image_url}
                alt="Базовая генерация без оптимизации"
              />
            </div>

            <div className="vc-frame__meta">
              <span className="param-pill mono">
                <span>g:</span>
                <strong>{rendered.baseline.guidance_scale.toFixed(1)}</strong>
              </span>
              <span className="param-pill mono">
                <span>steps:</span>
                <strong>{rendered.baseline.steps}</strong>
              </span>
              <span className="param-pill mono">
                <span>рендер:</span>
                <strong>{rendered.baseline.render_time_sec.toFixed(1)} с</strong>
              </span>
            </div>

            <p className="vc-frame__prompt mono">{rendered.baseline.prompt}</p>
          </article>

          <article className="vc-frame vc-frame--winner">
            <div className="vc-frame__head">
              <span className="vc-frame__title">После · ConvergeLab</span>
              <span
                className="pill-badge mono"
                style={{
                  backgroundColor: 'var(--status-success-soft)',
                  color: 'var(--status-success)',
                }}
              >
                F = {bestEval.fitness.toFixed(4)} ({gainPct >= 0 ? '+' : ''}
                {gainPct.toFixed(1)}%)
              </span>
            </div>

            <div className="vc-frame__media">
              <img
                className="vc-frame__img"
                src={rendered.optimized.image_url}
                alt="Оптимизированная генерация ConvergeLab"
              />
            </div>

            <div className="vc-frame__meta">
              <span className="param-pill mono">
                <span>g:</span>
                <strong>{rendered.optimized.guidance_scale.toFixed(2)}</strong>
              </span>
              <span className="param-pill mono">
                <span>steps:</span>
                <strong>{rendered.optimized.steps}</strong>
              </span>
              <span className="param-pill mono">
                <span>рендер:</span>
                <strong>{rendered.optimized.render_time_sec.toFixed(1)} с</strong>
              </span>
            </div>

            <p className="vc-frame__prompt mono">{rendered.optimized.prompt}</p>
          </article>
        </div>
      ) : (
        <div className="vc-slider-layout">
          <div
            ref={sliderStageRef}
            className="vc-slider-stage"
            role="slider"
            tabIndex={0}
            aria-label="Шторка сравнения До и После"
            aria-valuemin={4}
            aria-valuemax={96}
            aria-valuenow={Math.round(sliderPos)}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerCancel={handlePointerUp}
            onKeyDown={handleSliderKeyDown}
          >
            <img
              className="vc-slider-img"
              src={rendered.optimized.image_url}
              alt="После оптимизации"
              draggable={false}
            />
            <img
              className="vc-slider-img"
              src={rendered.baseline.image_url}
              alt="До оптимизации"
              draggable={false}
              style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}
            />

            <span className="vc-slider-tag vc-slider-tag--left mono">
              До · F₀ {result.macro_metrics.baseline_fitness.toFixed(3)}
            </span>
            <span className="vc-slider-tag vc-slider-tag--right mono">
              После · F {bestEval.fitness.toFixed(3)}
            </span>

            <div
              className="vc-slider-divider"
              style={{ left: `${sliderPos}%` }}
              aria-hidden="true"
            >
              <span className="vc-slider-handle">
                <MoveHorizontal size={18} />
              </span>
            </div>
          </div>

          <p className="vc-slider-hint">
            Потяни ползунок влево-вправо (или используй стрелки ← → на клавиатуре)
          </p>
        </div>
      )}

      {rendered && (
        <div className="vc-footer">
                    <p className="vc-footnote">
            Визуальный рендер найденной конфигурации (модель:{' '}
            <span className="mono">{rendered.model_id}</span>, единый seed).
            Числовые метрики F₀ и F_best рассчитаны суррогатным ядром стенда.
          </p>
          <span className="param-pill mono">
            <span>seed:</span>
            <strong>{rendered.seed}</strong>
            <span>·</span>
            <span>{rendered.total_wall_time_sec.toFixed(1)} с</span>
          </span>
        </div>
      )}
    </section>
  );
}
