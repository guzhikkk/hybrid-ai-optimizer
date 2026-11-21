import { useMemo } from 'react';
import type { KeyboardEvent } from 'react';
import { Award, BarChart3, Loader2, Play, Zap } from 'lucide-react';
import { EXAMPLE_PROMPTS } from '../services/apiClient';
import { ConvergenceChart } from './ConvergenceChart';
import type { ChartSeries } from './ConvergenceChart';
import type { ComparisonResult, ComparisonSummaryRow } from '../types/api';


interface CompareViewProps {
  corePrompt: string;
  onCorePromptChange: (prompt: string) => void;
  numRuns: number;
  onNumRunsChange: (runs: number) => void;
  maxIterations: number;
  onMaxIterationsChange: (iters: number) => void;
  targetFitness: number;
  onTargetFitnessChange: (val: number) => void;
  comparison: ComparisonResult | null;
  isLoading: boolean;
  onRunCompare: () => void;
}


const PRESET_COLORS: Record<string, string> = {
  pso_neuro_ddim: 'var(--series-pso)',
  ga_neuro_ddim: 'var(--series-ga)',
  random_neuro_ddim: 'var(--series-random-ddim)',
  baseline_random_zeroshot_ddpm: 'var(--series-baseline)',
};


const PRESET_SHORT_NAMES: Record<string, string> = {
  pso_neuro_ddim: 'Hybrid PSO + Нейросимволика + DDIM',
  ga_neuro_ddim: 'GA + Нейросимволика + DDIM',
  random_neuro_ddim: 'Random + Нейросимволика + DDIM',
  baseline_random_zeroshot_ddpm: 'Random + Zero-shot + DDPM',
};


function getPresetColor(presetId: string): string {
  return PRESET_COLORS[presetId] ?? 'var(--accent-blue)';
}


export function CompareView({
  corePrompt,
  onCorePromptChange,
  numRuns,
  onNumRunsChange,
  maxIterations,
  onMaxIterationsChange,
  targetFitness,
  onTargetFitnessChange,
  comparison,
  isLoading,
  onRunCompare,
}: CompareViewProps) {
  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      if (!isLoading && corePrompt.trim()) {
        onRunCompare();
      }
    }
  };


  const chartSeries: ChartSeries[] = useMemo(() => {
    if (!comparison) return [];
    return comparison.presets.map((preset) => ({
      id: preset.preset_id,
      label: PRESET_SHORT_NAMES[preset.preset_id] ?? preset.label,
      color: getPresetColor(preset.preset_id),
      dashed: preset.preset_id === 'baseline_random_zeroshot_ddpm',
      areaFill: preset.preset_id === comparison.winner_preset_id,
      values: preset.convergence_curve.map((pt) => ({
        iteration: pt.iteration,
        value: pt.global_best_fitness,
      })),
    }));
  }, [comparison]);


  const tableStats = useMemo(() => {
    if (!comparison || comparison.summary_table.length === 0) {
      return null;
    }
    const rows = comparison.summary_table;
    const convValues = rows
      .map((r) => r.mean_convergence_iteration)
      .filter((v): v is number => v !== null);


    return {
      bestFitness: Math.max(...rows.map((r) => r.mean_final_fitness)),
      bestGain: Math.max(...rows.map((r) => r.mean_relative_gain_pct)),
      bestConv: convValues.length > 0 ? Math.min(...convValues) : null,
      bestTime: Math.min(...rows.map((r) => r.mean_total_time_sec)),
      bestStd: Math.min(...rows.map((r) => r.fitness_std)),
      bestLpips: Math.min(...rows.map((r) => r.mean_lpips)),
    };
  }, [comparison]);


  const winnerRow: ComparisonSummaryRow | undefined = comparison?.summary_table.find(
    (r) => r.preset_id === comparison.winner_preset_id,
  );
  const baselineRow: ComparisonSummaryRow | undefined = comparison?.summary_table.find(
    (r) => r.preset_id === 'baseline_random_zeroshot_ddpm',
  );
  const randomDdimRow: ComparisonSummaryRow | undefined = comparison?.summary_table.find(
    (r) => r.preset_id === 'random_neuro_ddim',
  );


  const fitnessDelta =
    winnerRow && baselineRow
      ? winnerRow.mean_final_fitness - baselineRow.mean_final_fitness
      : null;


  const lpipsFactor =
    winnerRow && baselineRow && winnerRow.mean_lpips > 0
      ? baselineRow.mean_lpips / winnerRow.mean_lpips
      : null;


  const maxLpips = Math.max(
    0.01,
    baselineRow?.mean_lpips ?? 0,
    randomDdimRow?.mean_lpips ?? 0,
    winnerRow?.mean_lpips ?? 0,
  );


  const maxTime = Math.max(
    1,
    baselineRow?.mean_total_time_sec ?? 0,
    randomDdimRow?.mean_total_time_sec ?? 0,
    winnerRow?.mean_total_time_sec ?? 0,
  );


  return (
    <div className="compare-stack">
      {/* Control island */}
      <section className="prompt-bar" aria-label="Настройки сравнительного эксперимента">
        <div className="prompt-island">
          <div className="prompt-island__row">
            <textarea
              className="prompt-island__input"
              rows={2}
              value={corePrompt}
              onChange={(e) => onCorePromptChange(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Введите запрос T₀ для сравнительного бенчмарка всех 4 конфигураций..."
              aria-label="Запрос для сравнительного анализа"
            />


            <button
              type="button"
              className="btn-primary"
              onClick={onRunCompare}
              disabled={isLoading || !corePrompt.trim()}
            >
              {isLoading ? (
                <>
                  <Loader2 size={16} className="spin-icon" />
                  <span>Замер...</span>
                </>
              ) : (
                <>
                  <Play size={16} />
                  <span>Сравнить 4 связки</span>
                </>
              )}
            </button>
          </div>


          <div className="prompt-island__footer">
            <div className="prompt-chips">
              {EXAMPLE_PROMPTS.map((item) => (
                <button
                  key={item.label}
                  type="button"
                  className="prompt-chip"
                  onClick={() => onCorePromptChange(item.prompt)}
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>
        </div>


        <div className="controls-bar">
          <label className="select-pill">
            <span className="select-pill__label">Повторов (M для σ_F)</span>
            <select
              className="select-pill__field mono"
              value={numRuns}
              onChange={(e) => onNumRunsChange(Number(e.target.value))}
            >
              <option value={1}>1 прогон</option>
              <option value={3}>3 прогона</option>
              <option value={5}>5 прогонов</option>
            </select>
          </label>


          <label className="select-pill">
            <span className="select-pill__label">Итераций</span>
            <select
              className="select-pill__field mono"
              value={maxIterations}
              onChange={(e) => onMaxIterationsChange(Number(e.target.value))}
            >
              <option value={8}>8</option>
              <option value={12}>12</option>
              <option value={15}>15</option>
            </select>
          </label>


          <label className="select-pill">
            <span className="select-pill__label">Порог цели F</span>
            <select
              className="select-pill__field mono"
              value={targetFitness.toFixed(2)}
              onChange={(e) => onTargetFitnessChange(Number(e.target.value))}
            >
              <option value="0.85">0.85</option>
              <option value="0.88">0.88</option>
              <option value="0.90">0.90</option>
            </select>
          </label>
        </div>
      </section>


      {!comparison ? (
        <section className="card">
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 'var(--space-3)',
              padding: 'var(--space-10) var(--space-4)',
              textAlign: 'center',
            }}
          >
            <span className="empty-overlay__icon">
              <BarChart3 size={20} />
            </span>
            <h2 className="card-title">Сравнительный анализ 4 конфигураций</h2>
            <p className="card-subtitle" style={{ maxWidth: '480px', fontSize: '14px', lineHeight: '20px' }}>
              {isLoading
                ? 'Выполняются повторные прогоны Random Search, GA и Hybrid PSO на едином запросе...'
                : 'Нажмите «Сравнить 4 связки», чтобы прогнать базовый DDPM + Zero-shot против Random, Genetic Algorithm и Hybrid PSO с Нейросимволикой и DDIM.'}
            </p>
          </div>
        </section>
      ) : (
        <>
          {/* Hero Winner Banner */}
          <section className="winner-banner" aria-label="Лучшая конфигурация">
            <div className="winner-banner__left">
              <span className="winner-banner__icon">
                <Award size={24} />
              </span>
              <div>
                <p className="winner-banner__eyebrow">
                  Победитель бенчмарка ({comparison.num_runs} повтор. прогон.)
                </p>
                <h2 className="winner-banner__title">{comparison.winner_label}</h2>
              </div>
            </div>


            <div className="winner-stats">
              <div className="winner-stat">
                <span className="winner-stat__label">Преимущество vs Бейзлайн</span>
                <span
                  className="winner-stat__value mono"
                  style={{ color: 'var(--status-success)' }}
                >
                  {fitnessDelta !== null
                    ? `+${fitnessDelta.toFixed(4)} F`
                    : '—'}
                </span>
              </div>


              <div className="winner-stat">
                <span className="winner-stat__label">Сходимость N_conv</span>
                <span className="winner-stat__value mono">
                  {winnerRow?.mean_convergence_iteration !== null &&
                  winnerRow?.mean_convergence_iteration !== undefined
                    ? `Шаг ${winnerRow.mean_convergence_iteration.toFixed(1)}`
                    : 'Ниже порога'}
                </span>
              </div>


              <div className="winner-stat">
                <span className="winner-stat__label">Снижение шума LPIPS</span>
                <span
                  className="winner-stat__value mono"
                  style={{ color: 'var(--accent-blue)' }}
                >
                  {lpipsFactor !== null ? `${lpipsFactor.toFixed(1)}× ниже` : '—'}
                </span>
              </div>
            </div>
          </section>


          {/* Multi-curve convergence chart */}
          <ConvergenceChart
            title="Сравнение кривых сходимости F(t)"
            subtitle="Динамика глобального рекорда по итерациям для всех 4 исследуемых связок"
            series={chartSeries}
            targetFitness={comparison.target_fitness}
            isLoading={isLoading}
          />


          {/* Summary metrics table */}
          <section className="card" aria-label="Сводная таблица метрик">
            <div className="card-header">
              <div>
                <h2 className="card-title">Сводная таблица макро-метрик (Уровень 2)</h2>
                <p className="card-subtitle">
                  Усреднение по {comparison.num_runs} независимым запускам · лучшие значения в столбцах подсвечены
                </p>
              </div>
            </div>


            <div className="table-scroll">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Конфигурация</th>
                    <th className="num-col">F_best</th>
                    <th className="num-col">Прирост ΔF</th>
                    <th className="num-col">Сходимость N_conv</th>
                    <th className="num-col">Время (с)</th>
                    <th className="num-col">Разброс σ_F</th>
                    <th className="num-col">LPIPS</th>
                  </tr>
                </thead>
                <tbody>
                  {comparison.summary_table.map((row) => {
                    const isWinner = row.preset_id === comparison.winner_preset_id;
                    const isBestF =
                      tableStats !== null &&
                      row.mean_final_fitness === tableStats.bestFitness;
                    const isBestGain =
                      tableStats !== null &&
                      row.mean_relative_gain_pct === tableStats.bestGain;
                    const isBestConv =
                      tableStats !== null &&
                      tableStats.bestConv !== null &&
                      row.mean_convergence_iteration === tableStats.bestConv;
                    const isBestTime =
                      tableStats !== null &&
                      row.mean_total_time_sec === tableStats.bestTime;
                    const isBestStd =
                      tableStats !== null &&
                      row.fitness_std === tableStats.bestStd;
                    const isBestLpips =
                      tableStats !== null &&
                      row.mean_lpips === tableStats.bestLpips;


                    return (
                      <tr
                        key={row.preset_id}
                        className={isWinner ? 'row-winner' : ''}
                      >
                        <td>
                          <div className="preset-cell">
                            <span
                              className="preset-dot"
                              style={{ backgroundColor: getPresetColor(row.preset_id) }}
                            />
                            <span className="preset-name">{row.label}</span>
                            {isWinner && <span className="leader-chip">Лидер</span>}
                          </div>
                        </td>
                        <td className="num-col mono">
                          <span className={isBestF ? 'cell-best' : ''}>
                            {row.mean_final_fitness.toFixed(4)}
                          </span>
                        </td>
                        <td className="num-col mono">
                          <span className={isBestGain ? 'cell-best' : ''}>
                            {row.mean_relative_gain_pct >= 0 ? '+' : ''}
                            {row.mean_relative_gain_pct.toFixed(1)}%
                          </span>
                        </td>
                        <td className="num-col mono">
                          <span className={isBestConv ? 'cell-best' : ''}>
                            {row.mean_convergence_iteration !== null
                              ? row.mean_convergence_iteration.toFixed(2)
                              : '—'}
                          </span>
                        </td>
                        <td className="num-col mono">
                          <span className={isBestTime ? 'cell-best' : ''}>
                            {row.mean_total_time_sec.toFixed(1)}
                          </span>
                        </td>
                        <td className="num-col mono">
                          <span className={isBestStd ? 'cell-best' : ''}>
                            {row.fitness_std.toFixed(4)}
                          </span>
                        </td>
                        <td className="num-col mono">
                          <span className={isBestLpips ? 'cell-best' : ''}>
                            {row.mean_lpips.toFixed(4)}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>


          {/* DDPM vs DDIM visual breakdown */}
          {baselineRow && winnerRow && (
            <section className="card" aria-label="Влияние планировщика DDPM против DDIM">
              <div className="card-header">
                <div>
                  <h2 className="card-title">Влияние планировщика: DDPM vs DDIM</h2>
                  <p className="card-subtitle">
                    Сравнение стохастического DDPM и детерминированного DDIM (η = 0) по перцептивной стабильности и затратам времени
                  </p>
                </div>
                <span
                  className="pill-badge"
                  style={{
                    backgroundColor: 'var(--accent-blue-soft)',
                    color: 'var(--accent-blue)',
                  }}
                >
                  <Zap size={14} />
                  <span>Узел 3 · FC-2.2</span>
                </span>
              </div>


              <div className="scheduler-grid">
                <div className="scheduler-panel">
                  <div>
                    <h3 className="scheduler-panel__title">
                      Перцептивный разброс LPIPS (чем ниже, тем лучше)
                    </h3>
                    <p className="scheduler-panel__desc">
                      Детерминизация траектории ODE в DDIM устраняет паразитный шум отклика
                    </p>
                  </div>


                  <div className="meter-stack">
                    <div className="meter-item">
                      <div className="meter-item__head">
                        <span className="meter-item__label">
                          DDPM · Random + Zero-shot
                        </span>
                        <span className="meter-item__value mono">
                          {baselineRow.mean_lpips.toFixed(4)}
                        </span>
                      </div>
                      <div className="meter-track">
                        <div
                          className="meter-fill meter-fill--error"
                          style={{
                            width: `${Math.min(100, (baselineRow.mean_lpips / maxLpips) * 100)}%`,
                          }}
                        />
                      </div>
                    </div>


                    {randomDdimRow && (
                      <div className="meter-item">
                        <div className="meter-item__head">
                          <span className="meter-item__label">
                            DDIM · Random + Нейросимволика
                          </span>
                          <span className="meter-item__value mono">
                            {randomDdimRow.mean_lpips.toFixed(4)}
                          </span>
                        </div>
                        <div className="meter-track">
                          <div
                            className="meter-fill meter-fill--warning"
                            style={{
                              width: `${Math.min(100, (randomDdimRow.mean_lpips / maxLpips) * 100)}%`,
                            }}
                          />
                        </div>
                      </div>
                    )}


                    <div className="meter-item">
                      <div className="meter-item__head">
                        <span className="meter-item__label">
                          DDIM · Hybrid PSO + Нейросимволика
                        </span>
                        <span className="meter-item__value mono">
                          {winnerRow.mean_lpips.toFixed(4)}
                        </span>
                      </div>
                      <div className="meter-track">
                        <div
                          className="meter-fill meter-fill--green"
                          style={{
                            width: `${Math.min(100, (winnerRow.mean_lpips / maxLpips) * 100)}%`,
                          }}
                        />
                      </div>
                    </div>
                  </div>
                </div>


                <div className="scheduler-panel">
                  <div>
                    <h3 className="scheduler-panel__title">
                      Вычислительный бюджет T_total (сек.)
                    </h3>
                    <p className="scheduler-panel__desc">
                      Суммарное расчётное время цикла оптимизации на одном запросе
                    </p>
                  </div>


                  <div className="meter-stack">
                    <div className="meter-item">
                      <div className="meter-item__head">
                        <span className="meter-item__label">
                          DDPM · Random + Zero-shot
                        </span>
                        <span className="meter-item__value mono">
                          {baselineRow.mean_total_time_sec.toFixed(1)} с
                        </span>
                      </div>
                      <div className="meter-track">
                        <div
                          className="meter-fill meter-fill--error"
                          style={{
                            width: `${Math.min(100, (baselineRow.mean_total_time_sec / maxTime) * 100)}%`,
                          }}
                        />
                      </div>
                    </div>


                    {randomDdimRow && (
                      <div className="meter-item">
                        <div className="meter-item__head">
                          <span className="meter-item__label">
                            DDIM · Random + Нейросимволика
                          </span>
                          <span className="meter-item__value mono">
                            {randomDdimRow.mean_total_time_sec.toFixed(1)} с
                          </span>
                        </div>
                        <div className="meter-track">
                          <div
                            className="meter-fill meter-fill--warning"
                            style={{
                              width: `${Math.min(100, (randomDdimRow.mean_total_time_sec / maxTime) * 100)}%`,
                            }}
                          />
                        </div>
                      </div>
                    )}


                    <div className="meter-item">
                      <div className="meter-item__head">
                        <span className="meter-item__label">
                          DDIM · Hybrid PSO + Нейросимволика
                        </span>
                        <span className="meter-item__value mono">
                          {winnerRow.mean_total_time_sec.toFixed(1)} с
                        </span>
                      </div>
                      <div className="meter-track">
                        <div
                          className="meter-fill meter-fill--blue"
                          style={{
                            width: `${Math.min(100, (winnerRow.mean_total_time_sec / maxTime) * 100)}%`,
                          }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </section>
          )}
        </>
      )}
    </div>
  );
}
