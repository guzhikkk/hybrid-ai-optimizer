import { useEffect, useRef, useState } from 'react';
import { Pause, Play, RotateCcw, Sparkles } from 'lucide-react';
import type { IterationLog, ThemeMode } from '../types/api';

type AxisMode = 'diffusion' | 'weights';

interface SwarmCanvasProps {
  history: IterationLog[];
  currentIndex: number;
  onIndexChange: (index: number) => void;
  isPlaying: boolean;
  onTogglePlay: () => void;
  onResetPlay: () => void;
  playbackSpeed: number;
  onSpeedChange: (speed: number) => void;
  theme: ThemeMode;
  isLoading: boolean;
}

interface AxisSpec {
  xIdx: number;
  yIdx: number;
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
  xLabel: string;
  yLabel: string;
  xDecimals: number;
  yDecimals: number;
}

const AXIS_SPECS: Record<AxisMode, AxisSpec> = {
  diffusion: {
    xIdx: 0,
    yIdx: 1,
    xMin: 3.0,
    xMax: 15.0,
    yMin: 10.0,
    yMax: 50.0,
    xLabel: 'Guidance Scale (g)',
    yLabel: 'Steps (s)',
    xDecimals: 1,
    yDecimals: 0,
  },
  weights: {
    xIdx: 2,
    yIdx: 3,
    xMin: 0.5,
    xMax: 1.5,
    yMin: 0.5,
    yMax: 1.5,
    xLabel: 'Вес стиля (w₁)',
    yLabel: 'Вес света (w₂)',
    xDecimals: 2,
    yDecimals: 2,
  },
};

const SPEED_OPTIONS = [1, 2, 4] as const;

function clamp(val: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, val));
}

export function SwarmCanvas({
  history,
  currentIndex,
  onIndexChange,
  isPlaying,
  onTogglePlay,
  onResetPlay,
  playbackSpeed,
  onSpeedChange,
  theme,
  isLoading,
}: SwarmCanvasProps) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [axisMode, setAxisMode] = useState<AxisMode>('diffusion');
  const [size, setSize] = useState({ width: 560, height: 320 });

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;

    const updateSize = () => {
      const rect = el.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        setSize({
          width: Math.round(rect.width),
          height: Math.round(rect.height),
        });
      }
    };

    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || history.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const safeIndex = clamp(currentIndex, 0, history.length - 1);
    const activeLog = history[safeIndex];
    const spec = AXIS_SPECS[axisMode];

    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(size.width * dpr);
    canvas.height = Math.round(size.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const styles = getComputedStyle(document.documentElement);
    const bgColor = styles.getPropertyValue('--canvas-bg').trim() || '#151618';
    const gridColor =
      styles.getPropertyValue('--chart-grid').trim() || 'rgba(227,227,227,0.08)';
    const textMuted =
      styles.getPropertyValue('--text-secondary').trim() || '#9aa0a6';
    const accentBlue =
      styles.getPropertyValue('--accent-blue').trim() || '#8ab4f8';
    const accentPurple =
      styles.getPropertyValue('--accent-purple').trim() || '#c58af9';
    const accentSuccess =
      styles.getPropertyValue('--status-success').trim() || '#81c995';

    const padLeft = 48;
    const padRight = 20;
    const padTop = 20;
    const padBottom = 40;
    const plotW = Math.max(1, size.width - padLeft - padRight);
    const plotH = Math.max(1, size.height - padTop - padBottom);

    const toX = (val: number) =>
      padLeft +
      ((clamp(val, spec.xMin, spec.xMax) - spec.xMin) / (spec.xMax - spec.xMin)) *
        plotW;
    const toY = (val: number) =>
      padTop +
      (1 -
        (clamp(val, spec.yMin, spec.yMax) - spec.yMin) / (spec.yMax - spec.yMin)) *
        plotH;

    // 1. Background
    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, size.width, size.height);

    // 2. Grid lines and axis labels
    const ticks = 4;
    ctx.strokeStyle = gridColor;
    ctx.lineWidth = 1;
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.fillStyle = textMuted;

    for (let i = 0; i <= ticks; i += 1) {
      const ratio = i / ticks;
      const gx = padLeft + ratio * plotW;
      const gy = padTop + (1 - ratio) * plotH;

      ctx.beginPath();
      ctx.moveTo(gx, padTop);
      ctx.lineTo(gx, padTop + plotH);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(padLeft, gy);
      ctx.lineTo(padLeft + plotW, gy);
      ctx.stroke();

      const xVal = spec.xMin + ratio * (spec.xMax - spec.xMin);
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(xVal.toFixed(spec.xDecimals), gx, padTop + plotH + 8);

      const yVal = spec.yMin + ratio * (spec.yMax - spec.yMin);
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(yVal.toFixed(spec.yDecimals), padLeft - 8, gy);
    }

    // Axis titles
    ctx.font = '500 11px "Inter", sans-serif';
    ctx.fillStyle = textMuted;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(spec.xLabel, padLeft + plotW / 2, size.height - 4);

    ctx.save();
    ctx.translate(12, padTop + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(spec.yLabel, 0, 0);
    ctx.restore();

    // 3. Soft glow zone around current Global Best
    const gBestPos = activeLog.global_best_position;
    const gBestX = toX(gBestPos[spec.xIdx] ?? spec.xMin);
    const gBestY = toY(gBestPos[spec.yIdx] ?? spec.yMin);

    const glow = ctx.createRadialGradient(gBestX, gBestY, 4, gBestX, gBestY, 56);
    glow.addColorStop(
      0,
      theme === 'dark' ? 'rgba(197, 138, 249, 0.24)' : 'rgba(132, 48, 206, 0.16)',
    );
    glow.addColorStop(1, 'rgba(197, 138, 249, 0)');
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(gBestX, gBestY, 56, 0, Math.PI * 2);
    ctx.fill();

    // 4. Recent particle trails (last 4 steps)
    const trailStart = Math.max(0, safeIndex - 4);
    for (let step = trailStart; step < safeIndex; step += 1) {
      const prevStep = history[step];
      const nextStep = history[step + 1];
      const alpha = 0.08 + ((step - trailStart + 1) / 5) * 0.22;

      const prevById = new Map(
        prevStep.particles.map((p) => [p.particle_id, p]),
      );

      ctx.strokeStyle = accentBlue;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = 1.5;

      for (const nextP of nextStep.particles) {
        const prevP = prevById.get(nextP.particle_id);
        if (!prevP) continue;
        const x1 = toX(prevP.position[spec.xIdx] ?? spec.xMin);
        const y1 = toY(prevP.position[spec.yIdx] ?? spec.yMin);
        const x2 = toX(nextP.position[spec.xIdx] ?? spec.xMin);
        const y2 = toY(nextP.position[spec.yIdx] ?? spec.yMin);

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;

    // 5. Active particles
    const bestFit = activeLog.global_best_fitness;
    for (const particle of activeLog.particles) {
      const px = toX(particle.position[spec.xIdx] ?? spec.xMin);
      const py = toY(particle.position[spec.yIdx] ?? spec.yMin);
      const fit = particle.current_evaluation.fitness;
      const isNearBest = bestFit - fit < 0.015;
      const radius = 4.5 + clamp(fit, 0, 1) * 2.5;

      ctx.beginPath();
      ctx.arc(px, py, radius + 4, 0, Math.PI * 2);
      ctx.fillStyle = isNearBest ? accentSuccess : accentBlue;
      ctx.globalAlpha = 0.18;
      ctx.fill();

      ctx.beginPath();
      ctx.arc(px, py, radius, 0, Math.PI * 2);
      ctx.globalAlpha = 0.95;
      ctx.fill();
    }
    ctx.globalAlpha = 1;

    // 6. Global Best (G_best) marker
    ctx.beginPath();
    ctx.arc(gBestX, gBestY, 12, 0, Math.PI * 2);
    ctx.strokeStyle = accentPurple;
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(gBestX, gBestY, 5, 0, Math.PI * 2);
    ctx.fillStyle = accentPurple;
    ctx.fill();

    // Label for G_best (keep inside plot bounds)
    const labelText = `G_best (${bestFit.toFixed(3)})`;
    ctx.font = '600 11px "JetBrains Mono", monospace';
    const textW = ctx.measureText(labelText).width;
    const labelX =
      gBestX + 16 + textW > padLeft + plotW ? gBestX - 16 - textW : gBestX + 16;
    const labelY = gBestY - 14 < padTop + 8 ? gBestY + 14 : gBestY - 10;

    ctx.fillStyle = accentPurple;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(labelText, labelX, labelY);
  }, [history, currentIndex, axisMode, size, theme]);

  const maxIdx = Math.max(0, history.length - 1);
  const hasData = history.length > 0;
  const safeIdx = clamp(currentIndex, 0, maxIdx);

  const cycleSpeed = () => {
    const idx = SPEED_OPTIONS.indexOf(
      playbackSpeed as (typeof SPEED_OPTIONS)[number],
    );
    const next = SPEED_OPTIONS[(idx + 1) % SPEED_OPTIONS.length];
    onSpeedChange(next);
  };

  return (
    <article className="card">
      <div className="card-header">
        <div>
          <h2 className="card-title">Пространство поиска роя</h2>
          <p className="card-subtitle">
            Траектории частиц и схождение к глобальному рекорду G_best
          </p>
        </div>

        <div className="axis-toggle" role="group" aria-label="Проекция осей">
          <button
            type="button"
            className={`axis-toggle__btn mono ${
              axisMode === 'diffusion' ? 'axis-toggle__btn--active' : ''
            }`}
            onClick={() => setAxisMode('diffusion')}
          >
            g × s
          </button>
          <button
            type="button"
            className={`axis-toggle__btn mono ${
              axisMode === 'weights' ? 'axis-toggle__btn--active' : ''
            }`}
            onClick={() => setAxisMode('weights')}
          >
            w₁ × w₂
          </button>
        </div>
      </div>

      <div className="swarm-canvas-wrap" ref={wrapRef}>
        <canvas ref={canvasRef} className="swarm-canvas" />
        {!hasData && (
          <div className="empty-overlay">
            <span className="empty-overlay__icon">
              <Sparkles size={20} />
            </span>
            <p className="empty-overlay__text">
              {isLoading
                ? 'Выполняется расчёт траекторий роя...'
                : 'Нажмите «Запустить», чтобы увидеть полёт частиц к оптимуму'}
            </p>
          </div>
        )}
      </div>

      <div className="player-bar">
        <button
          type="button"
          className="player-btn player-btn--primary"
          onClick={onTogglePlay}
          disabled={!hasData}
          aria-label={isPlaying ? 'Пауза' : 'Воспроизвести'}
        >
          {isPlaying ? <Pause size={16} /> : <Play size={16} />}
        </button>

        <button
          type="button"
          className="player-btn"
          onClick={onResetPlay}
          disabled={!hasData}
          aria-label="С начала"
          title="В начало (итерация 0)"
        >
          <RotateCcw size={16} />
        </button>

        <input
          type="range"
          className="player-slider"
          min={0}
          max={maxIdx}
          step={1}
          value={safeIdx}
          disabled={!hasData}
          onChange={(e) => onIndexChange(Number(e.target.value))}
          aria-label="Текущая итерация"
        />

        <span className="player-counter mono">
          {hasData ? `Шаг ${safeIdx} / ${maxIdx}` : 'Шаг 0 / 0'}
        </span>

        <button
          type="button"
          className="speed-btn mono"
          onClick={cycleSpeed}
          disabled={!hasData}
          title="Скорость воспроизведения"
        >
          {playbackSpeed}x
        </button>
      </div>
    </article>
  );
}
