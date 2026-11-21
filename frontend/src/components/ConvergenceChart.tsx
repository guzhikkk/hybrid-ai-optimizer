import { useEffect, useId, useRef, useState } from 'react';
import type { MouseEvent } from 'react';
import { TrendingUp } from 'lucide-react';

export interface ChartPoint {
  iteration: number;
  value: number;
}

export interface ChartSeries {
  id: string;
  label: string;
  color: string;
  values: ChartPoint[];
  dashed?: boolean;
  areaFill?: boolean;
}

interface ConvergenceChartProps {
  title: string;
  subtitle: string;
  series: ChartSeries[];
  targetFitness?: number;
  activeIteration?: number | null;
  onSelectIteration?: (iteration: number) => void;
  isLoading?: boolean;
}

export function ConvergenceChart({
  title,
  subtitle,
  series,
  targetFitness,
  activeIteration = null,
  onSelectIteration,
  isLoading = false,
}: ConvergenceChartProps) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const gradPrefix = useId().replace(/:/g, '');
  const [size, setSize] = useState({ width: 560, height: 320 });
  const [hoveredIter, setHoveredIter] = useState<number | null>(null);

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

  const nonEmptySeries = series.filter((s) => s.values.length > 0);
  const hasData = nonEmptySeries.length > 0;

  // Domain calculation
  const allValues = nonEmptySeries.flatMap((s) => s.values.map((p) => p.value));
  if (typeof targetFitness === 'number' && hasData) {
    allValues.push(targetFitness);
  }

  const rawMin = allValues.length > 0 ? Math.min(...allValues) : 0.4;
  const rawMax = allValues.length > 0 ? Math.max(...allValues) : 1.0;
  const yMin = Math.max(0, Math.floor((rawMin - 0.04) * 20) / 20);
  const yMaxCandidate = Math.min(1, Math.ceil((rawMax + 0.03) * 20) / 20);
  const yMax = yMaxCandidate - yMin < 0.1 ? Math.min(1, yMin + 0.15) : yMaxCandidate;

  const allIters = nonEmptySeries.flatMap((s) => s.values.map((p) => p.iteration));
  const maxIter = allIters.length > 0 ? Math.max(1, ...allIters) : 1;

  const padLeft = 48;
  const padRight = 20;
  const padTop = 20;
  const padBottom = 36;
  const plotW = Math.max(1, size.width - padLeft - padRight);
  const plotH = Math.max(1, size.height - padTop - padBottom);

  const toX = (iter: number) => padLeft + (iter / maxIter) * plotW;
  const toY = (val: number) => {
    const ratio = (val - yMin) / Math.max(1e-6, yMax - yMin);
    return padTop + (1 - Math.max(0, Math.min(1, ratio))) * plotH;
  };

  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((t) => yMin + t * (yMax - yMin));
  const xTickStep = maxIter <= 8 ? 1 : maxIter <= 16 ? 2 : 5;
  const xTicks: number[] = [];
  for (let i = 0; i <= maxIter; i += xTickStep) {
    xTicks.push(i);
  }
  if (xTicks[xTicks.length - 1] !== maxIter) {
    xTicks.push(maxIter);
  }

  const buildPolyline = (points: ChartPoint[]) =>
    points.map((p) => `${toX(p.iteration).toFixed(1)},${toY(p.value).toFixed(1)}`).join(' ');

  const buildAreaPath = (points: ChartPoint[]) => {
    if (points.length === 0) return '';
    const first = points[0];
    const last = points[points.length - 1];
    const linePart = points
      .map(
        (p, idx) =>
          `${idx === 0 ? 'M' : 'L'} ${toX(p.iteration).toFixed(1)} ${toY(p.value).toFixed(1)}`,
      )
      .join(' ');
    const baseY = (padTop + plotH).toFixed(1);
    return `${linePart} L ${toX(last.iteration).toFixed(1)} ${baseY} L ${toX(first.iteration).toFixed(1)} ${baseY} Z`;
  };

  const handleMouseMove = (event: MouseEvent<SVGRectElement>) => {
    if (!hasData) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const relX = event.clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, relX / Math.max(1, rect.width)));
    setHoveredIter(Math.round(ratio * maxIter));
  };

  const handleChartClick = () => {
    if (hoveredIter !== null && onSelectIteration) {
      onSelectIteration(hoveredIter);
    }
  };

  const focusIter = hoveredIter ?? activeIteration;
  const tooltipOnLeft =
    hoveredIter !== null && toX(hoveredIter) > padLeft + plotW * 0.58;

  return (
    <article className="card">
      <div className="card-header">
        <div>
          <h2 className="card-title">{title}</h2>
          <p className="card-subtitle">{subtitle}</p>
        </div>
        {typeof targetFitness === 'number' && (
          <span className="pill-badge mono" style={{ background: 'var(--bg-surface-high)', color: 'var(--text-secondary)' }}>
            Цель F ≥ {targetFitness.toFixed(2)}
          </span>
        )}
      </div>

      <div className="chart-wrap" ref={wrapRef}>
        {hasData ? (
          <>
            <svg
              className="chart-svg"
              width={size.width}
              height={size.height}
              viewBox={`0 0 ${size.width} ${size.height}`}
              role="img"
              aria-label={title}
            >
              <defs>
                {nonEmptySeries.map((s, idx) =>
                  s.areaFill ? (
                    <linearGradient
                      key={s.id}
                      id={`${gradPrefix}-area-${idx}`}
                      x1="0"
                      y1="0"
                      x2="0"
                      y2="1"
                    >
                      <stop offset="0%" stopColor={s.color} stopOpacity={0.28} />
                      <stop offset="100%" stopColor={s.color} stopOpacity={0.0} />
                    </linearGradient>
                  ) : null,
                )}
              </defs>

              {/* Horizontal grid & Y labels */}
              {yTicks.map((val) => {
                const y = toY(val);
                return (
                  <g key={val.toFixed(4)}>
                    <line
                      x1={padLeft}
                      y1={y}
                      x2={padLeft + plotW}
                      y2={y}
                      stroke="var(--chart-grid)"
                      strokeWidth={1}
                    />
                    <text
                      x={padLeft - 8}
                      y={y}
                      textAnchor="end"
                      dominantBaseline="middle"
                      fill="var(--text-secondary)"
                      fontSize={11}
                      fontFamily="var(--font-mono)"
                    >
                      {val.toFixed(2)}
                    </text>
                  </g>
                );
              })}

              {/* X labels */}
              {xTicks.map((iter) => {
                const x = toX(iter);
                return (
                  <text
                    key={iter}
                    x={x}
                    y={padTop + plotH + 16}
                    textAnchor="middle"
                    dominantBaseline="hanging"
                    fill="var(--text-secondary)"
                    fontSize={11}
                    fontFamily="var(--font-mono)"
                  >
                    {iter}
                  </text>
                );
              })}

              {/* Target fitness threshold */}
              {typeof targetFitness === 'number' &&
                targetFitness >= yMin &&
                targetFitness <= yMax && (
                  <line
                    x1={padLeft}
                    y1={toY(targetFitness)}
                    x2={padLeft + plotW}
                    y2={toY(targetFitness)}
                    stroke="var(--status-success)"
                    strokeWidth={1.5}
                    strokeDasharray="6 4"
                    opacity={0.75}
                  />
                )}

              {/* Area fills */}
              {nonEmptySeries.map((s, idx) =>
                s.areaFill ? (
                  <path
                    key={`${s.id}-area`}
                    d={buildAreaPath(s.values)}
                    fill={`url(#${gradPrefix}-area-${idx})`}
                  />
                ) : null,
              )}

              {/* Series polylines */}
              {nonEmptySeries.map((s) => (
                <polyline
                  key={s.id}
                  fill="none"
                  stroke={s.color}
                  strokeWidth={s.dashed ? 1.75 : 2.5}
                  strokeDasharray={s.dashed ? '5 4' : undefined}
                  strokeLinejoin="round"
                  strokeLinecap="round"
                  points={buildPolyline(s.values)}
                />
              ))}

              {/* Active / Hovered vertical cursor & highlight dots */}
              {focusIter !== null && (
                <g>
                  <line
                    x1={toX(focusIter)}
                    y1={padTop}
                    x2={toX(focusIter)}
                    y2={padTop + plotH}
                    stroke="var(--border-strong)"
                    strokeWidth={1}
                    strokeDasharray="4 4"
                  />
                  {nonEmptySeries.map((s) => {
                    const pt = s.values.find((p) => p.iteration === focusIter);
                    if (!pt) return null;
                    return (
                      <circle
                        key={`${s.id}-dot`}
                        cx={toX(pt.iteration)}
                        cy={toY(pt.value)}
                        r={4.5}
                        fill={s.color}
                        stroke="var(--bg-surface)"
                        strokeWidth={2}
                      />
                    );
                  })}
                </g>
              )}

              {/* Interactive hit area */}
              <rect
                x={padLeft}
                y={padTop}
                width={plotW}
                height={plotH}
                fill="transparent"
                style={{ cursor: onSelectIteration ? 'pointer' : 'crosshair' }}
                onMouseMove={handleMouseMove}
                onMouseLeave={() => setHoveredIter(null)}
                onClick={handleChartClick}
              />
            </svg>

            {hoveredIter !== null && (
              <div
                className="chart-tooltip"
                style={
                  tooltipOnLeft
                    ? { left: padLeft + 12 }
                    : { right: padRight + 12 }
                }
              >
                <p className="chart-tooltip__title mono">
                  Итерация {hoveredIter}
                </p>
                {nonEmptySeries.map((s) => {
                  const pt = s.values.find((p) => p.iteration === hoveredIter);
                  if (!pt) return null;
                  return (
                    <div key={s.id} className="chart-tooltip__row">
                      <span className="chart-tooltip__name">
                        <span
                          className="chart-tooltip__dot"
                          style={{ backgroundColor: s.color }}
                        />
                        <span>{s.label}</span>
                      </span>
                      <span className="chart-tooltip__val mono">
                        {pt.value.toFixed(4)}
                      </span>
                    </div>
                  );
                })}
              </div>
            )}
          </>
        ) : (
          <div className="empty-overlay" style={{ borderRadius: 'var(--radius-md)' }}>
            <span className="empty-overlay__icon">
              <TrendingUp size={20} />
            </span>
            <p className="empty-overlay__text">
              {isLoading
                ? 'Построение кривой сходимости...'
                : 'График сходимости появится сразу после запуска оптимизации'}
            </p>
          </div>
        )}
      </div>

      {hasData && (
        <div className="chart-legend">
          {nonEmptySeries.map((s) => (
            <div key={s.id} className="chart-legend__item">
              <span
                className={`chart-legend__swatch ${
                  s.dashed ? 'chart-legend__swatch--dashed' : ''
                }`}
                style={
                  s.dashed
                    ? { borderColor: s.color }
                    : { backgroundColor: s.color }
                }
              />
              <span>{s.label}</span>
            </div>
          ))}
        </div>
      )}
    </article>
  );
}
