import { Activity, Clock, Gauge, TrendingUp } from 'lucide-react';
import type { IterationLog, MacroMetrics, SchedulerType } from '../types/api';

interface KpiStripProps {
  metrics: MacroMetrics | null;
  currentLog: IterationLog | null;
  targetFitness: number;
  scheduler: SchedulerType;
  isLoading: boolean;
}

export function KpiStrip({
  metrics,
  currentLog,
  targetFitness,
  scheduler,
  isLoading,
}: KpiStripProps) {
  const activeFitness =
    currentLog?.global_best_fitness ?? metrics?.final_fitness ?? null;

  const gainPct = metrics?.relative_gain_pct ?? null;
  const lpips = metrics?.mean_lpips ?? null;
  const isSmooth = lpips !== null && lpips < 0.1;

  return (
    <section className="kpi-grid" aria-label="Ключевые показатели качества">
      <article className="kpi-card">
        <div className="kpi-card__head">
          <span className="kpi-card__label">Качество F_best</span>
          <span className="kpi-card__icon">
            <TrendingUp size={16} />
          </span>
        </div>
        <div className="kpi-card__body">
          {isLoading ? (
            <div className="skeleton-bar" />
          ) : (
            <>
              <span className="kpi-card__value mono">
                {activeFitness !== null ? activeFitness.toFixed(4) : '—'}
              </span>
              {gainPct !== null && (
                <span
                  className={`kpi-card__badge mono ${
                    gainPct >= 0
                      ? 'kpi-card__badge--success'
                      : 'kpi-card__badge--error'
                  }`}
                >
                  {gainPct >= 0 ? '+' : ''}
                  {gainPct.toFixed(1)}%
                </span>
              )}
            </>
          )}
        </div>
        <span className="kpi-card__foot mono">
          {metrics
            ? `Бейзлайн F₀: ${metrics.baseline_fitness.toFixed(4)}`
            : 'Ожидание запуска'}
        </span>
      </article>

      <article className="kpi-card">
        <div className="kpi-card__head">
          <span className="kpi-card__label">Сходимость N_conv</span>
          <span className="kpi-card__icon">
            <Gauge size={16} />
          </span>
        </div>
        <div className="kpi-card__body">
          {isLoading ? (
            <div className="skeleton-bar" />
          ) : (
            <>
              <span className="kpi-card__value mono">
                {metrics
                  ? metrics.convergence_iteration !== null
                    ? `Шаг ${metrics.convergence_iteration}`
                    : 'Нет'
                  : '—'}
              </span>
              {metrics && (
                <span
                  className={`kpi-card__badge ${
                    metrics.convergence_iteration !== null
                      ? 'kpi-card__badge--success'
                      : 'kpi-card__badge--warning'
                  }`}
                >
                  {metrics.convergence_iteration !== null
                    ? 'Цель достигнута'
                    : 'Ниже порога'}
                </span>
              )}
            </>
          )}
        </div>
        <span className="kpi-card__foot mono">
          Целевой порог F ≥ {targetFitness.toFixed(2)}
        </span>
      </article>

      <article className="kpi-card">
        <div className="kpi-card__head">
          <span className="kpi-card__label">Стабильность LPIPS</span>
          <span className="kpi-card__icon">
            <Activity size={16} />
          </span>
        </div>
        <div className="kpi-card__body">
          {isLoading ? (
            <div className="skeleton-bar" />
          ) : (
            <>
              <span className="kpi-card__value mono">
                {lpips !== null ? lpips.toFixed(4) : '—'}
              </span>
              {lpips !== null && (
                <span
                  className={`kpi-card__badge ${
                    isSmooth
                      ? 'kpi-card__badge--success'
                      : 'kpi-card__badge--error'
                  }`}
                >
                  {isSmooth ? 'Гладкий' : 'Шумный'}
                </span>
              )}
            </>
          )}
        </div>
        <span className="kpi-card__foot">
          {scheduler === 'ddim'
            ? 'DDIM · Детерминированный (η = 0)'
            : 'DDPM · Стохастический шум'}
        </span>
      </article>

      <article className="kpi-card">
        <div className="kpi-card__head">
          <span className="kpi-card__label">Расчётное время GPU</span>
          <span className="kpi-card__icon">
            <Clock size={16} />
          </span>
        </div>
        <div className="kpi-card__body">
          {isLoading ? (
            <div className="skeleton-bar" />
          ) : (
            <>
              <span className="kpi-card__value mono">
                {metrics ? `${metrics.total_time_sec.toFixed(1)} с` : '—'}
              </span>
              {metrics && (
                <span className="kpi-card__badge kpi-card__badge--neutral mono">
                  σ = {metrics.fitness_std.toFixed(4)}
                </span>
              )}
            </>
          )}
        </div>
        <span className="kpi-card__foot">
          {currentLog
            ? `Кадр итерации ${currentLog.iteration}`
            : 'Суммарно по циклу'}
        </span>
      </article>
    </section>
  );
}