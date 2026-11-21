import { useState } from 'react';
import type { KeyboardEvent } from 'react';
import { Loader2, Play, SlidersHorizontal } from 'lucide-react';
import { EXAMPLE_PROMPTS } from '../services/apiClient';
import type {
  ExperimentConfig,
  OptimizerType,
  PromptStrategyType,
  SchedulerType,
} from '../types/api';

interface PromptBarProps {
  config: ExperimentConfig;
  onChange: (patch: Partial<ExperimentConfig>) => void;
  onRun: () => void;
  isLoading: boolean;
}

export function PromptBar({
  config,
  onChange,
  onRun,
  isLoading,
}: PromptBarProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      if (!isLoading && config.core_prompt.trim()) {
        onRun();
      }
    }
  };

  const handleParticlesChange = (nextParticles: number) => {
    const safeElitism = Math.min(config.ga_elitism, nextParticles - 1);
    onChange({
      num_particles: nextParticles,
      ga_elitism: safeElitism,
    });
  };

  return (
    <section className="prompt-bar" aria-label="Параметры оптимизации">
      <div className="prompt-island">
        <div className="prompt-island__row">
          <textarea
            className="prompt-island__input"
            rows={2}
            value={config.core_prompt}
            onChange={(e) => onChange({ core_prompt: e.target.value })}
            onKeyDown={handleKeyDown}
            placeholder="Введите базовый запрос T₀, например: 3D mascot raccoon wearing a green hoodie..."
            aria-label="Базовый текстовый запрос"
          />

          <button
            type="button"
            className="btn-primary"
            onClick={onRun}
            disabled={isLoading || !config.core_prompt.trim()}
          >
            {isLoading ? (
              <>
                <Loader2 size={16} className="spin-icon" />
                <span>Поиск...</span>
              </>
            ) : (
              <>
                <Play size={16} />
                <span>Запустить</span>
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
                onClick={() => onChange({ core_prompt: item.prompt })}
              >
                {item.label}
              </button>
            ))}
          </div>

          <button
            type="button"
            className="btn-tonal"
            onClick={() => setShowAdvanced((prev) => !prev)}
            aria-expanded={showAdvanced}
          >
            <SlidersHorizontal size={16} />
            <span>{showAdvanced ? 'Скрыть параметры' : 'Параметры'}</span>
          </button>
        </div>
      </div>

      <div className="controls-bar">
        <label className="select-pill">
          <span className="select-pill__label">Алгоритм</span>
          <select
            className="select-pill__field"
            value={config.optimizer}
            onChange={(e) =>
              onChange({ optimizer: e.target.value as OptimizerType })
            }
          >
            <option value="pso">Hybrid PSO (Рекомендуемый)</option>
            <option value="ga">Genetic Algorithm (GA)</option>
            <option value="random">Random Search (Бейзлайн)</option>
          </select>
        </label>

        <label className="select-pill">
          <span className="select-pill__label">Стратегия LLM</span>
          <select
            className="select-pill__field"
            value={config.prompt_strategy}
            onChange={(e) =>
              onChange({ prompt_strategy: e.target.value as PromptStrategyType })
            }
          >
            <option value="neurosymbolic">Нейросимволика (Слоты + веса)</option>
            <option value="in_context">In-Context Learning (ICL)</option>
            <option value="zero_shot">Zero-shot (Свободная)</option>
          </select>
        </label>

        <label className="select-pill">
          <span className="select-pill__label">Планировщик</span>
          <select
            className="select-pill__field"
            value={config.scheduler}
            onChange={(e) =>
              onChange({ scheduler: e.target.value as SchedulerType })
            }
          >
            <option value="ddim">DDIM (Детерминированный)</option>
            <option value="ddpm">DDPM (Стохастический)</option>
          </select>
        </label>
      </div>

      {showAdvanced && (
        <div className="advanced-drawer">
          <div className="slider-field">
            <div className="slider-field__top">
              <span className="slider-field__label">Размер популяции (N)</span>
              <span className="slider-field__value mono">{config.num_particles}</span>
            </div>
            <input
              type="range"
              className="slider-field__input"
              min={4}
              max={20}
              step={1}
              value={config.num_particles}
              onChange={(e) => handleParticlesChange(Number(e.target.value))}
            />
          </div>

          <div className="slider-field">
            <div className="slider-field__top">
              <span className="slider-field__label">Итерации (I)</span>
              <span className="slider-field__value mono">{config.max_iterations}</span>
            </div>
            <input
              type="range"
              className="slider-field__input"
              min={4}
              max={30}
              step={1}
              value={config.max_iterations}
              onChange={(e) => onChange({ max_iterations: Number(e.target.value) })}
            />
          </div>

          <div className="slider-field">
            <div className="slider-field__top">
              <span className="slider-field__label">Целевой порог F</span>
              <span className="slider-field__value mono">
                {config.target_fitness.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              className="slider-field__input"
              min={0.7}
              max={0.95}
              step={0.01}
              value={config.target_fitness}
              onChange={(e) => onChange({ target_fitness: Number(e.target.value) })}
            />
          </div>

          {config.optimizer === 'pso' && (
            <>
              <div className="slider-field">
                <div className="slider-field__top">
                  <span className="slider-field__label">Инерция (ω)</span>
                  <span className="slider-field__value mono">
                    {config.pso_w.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  className="slider-field__input"
                  min={0.3}
                  max={1.0}
                  step={0.02}
                  value={config.pso_w}
                  onChange={(e) => onChange({ pso_w: Number(e.target.value) })}
                />
              </div>

              <div className="slider-field">
                <div className="slider-field__top">
                  <span className="slider-field__label">Когнитивный (c₁)</span>
                  <span className="slider-field__value mono">
                    {config.pso_c1.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  className="slider-field__input"
                  min={0.5}
                  max={2.5}
                  step={0.05}
                  value={config.pso_c1}
                  onChange={(e) => onChange({ pso_c1: Number(e.target.value) })}
                />
              </div>

              <div className="slider-field">
                <div className="slider-field__top">
                  <span className="slider-field__label">Социальный (c₂)</span>
                  <span className="slider-field__value mono">
                    {config.pso_c2.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  className="slider-field__input"
                  min={0.5}
                  max={2.5}
                  step={0.05}
                  value={config.pso_c2}
                  onChange={(e) => onChange({ pso_c2: Number(e.target.value) })}
                />
              </div>
            </>
          )}

          {config.optimizer === 'ga' && (
            <>
              <div className="slider-field">
                <div className="slider-field__top">
                  <span className="slider-field__label">Элитизм (особей)</span>
                  <span className="slider-field__value mono">{config.ga_elitism}</span>
                </div>
                <input
                  type="range"
                  className="slider-field__input"
                  min={0}
                  max={Math.min(4, config.num_particles - 1)}
                  step={1}
                  value={config.ga_elitism}
                  onChange={(e) => onChange({ ga_elitism: Number(e.target.value) })}
                />
              </div>

              <div className="slider-field">
                <div className="slider-field__top">
                  <span className="slider-field__label">Кроссовер (p_c)</span>
                  <span className="slider-field__value mono">
                    {config.ga_crossover_rate.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  className="slider-field__input"
                  min={0.2}
                  max={1.0}
                  step={0.05}
                  value={config.ga_crossover_rate}
                  onChange={(e) =>
                    onChange({ ga_crossover_rate: Number(e.target.value) })
                  }
                />
              </div>

              <div className="slider-field">
                <div className="slider-field__top">
                  <span className="slider-field__label">Мутация (p_m)</span>
                  <span className="slider-field__value mono">
                    {config.ga_mutation_rate.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  className="slider-field__input"
                  min={0.05}
                  max={0.5}
                  step={0.05}
                  value={config.ga_mutation_rate}
                  onChange={(e) =>
                    onChange({ ga_mutation_rate: Number(e.target.value) })
                  }
                />
              </div>
            </>
          )}
        </div>
      )}
    </section>
  );
}