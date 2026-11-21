import { useEffect, useRef, useState } from 'react';
import {
  AlertTriangle,
  Check,
  CheckCircle2,
  Copy,
  Sliders,
} from 'lucide-react';
import type { EvaluationBreakdown, PromptStrategyType } from '../types/api';

interface SolutionCardProps {
  evaluation: EvaluationBreakdown | null;
  maxTokens: number;
  strategy: PromptStrategyType;
  isLoading: boolean;
}

interface ParsedPrompt {
  core: string;
  slots: Array<{ text: string; weight: string }>;
}

const TOKEN_REGEX = /[a-zA-Zа-яА-ЯёЁ0-9]+/g;
const SLOT_REGEX = /\(([^()]+):(\d+(?:\.\d+)?)\)/g;

function countTokens(text: string): number {
  return text.match(TOKEN_REGEX)?.length ?? 0;
}

function parseNeurosymbolicPrompt(prompt: string): ParsedPrompt {
  const slots: Array<{ text: string; weight: string }> = [];
  let firstMatchIndex = -1;

  for (const match of prompt.matchAll(SLOT_REGEX)) {
    if (firstMatchIndex === -1 && typeof match.index === 'number') {
      firstMatchIndex = match.index;
    }
    slots.push({
      text: match[1].trim(),
      weight: match[2],
    });
  }

  if (slots.length === 0) {
    return { core: prompt, slots: [] };
  }

  const rawCore =
    firstMatchIndex > 0
      ? prompt.slice(0, firstMatchIndex).replace(/,\s*$/, '').trim()
      : '';

  return { core: rawCore, slots };
}

export function SolutionCard({
  evaluation,
  maxTokens,
  strategy,
  isLoading,
}: SolutionCardProps) {
  const [copied, setCopied] = useState(false);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current);
      }
    };
  }, []);

  const handleCopy = async () => {
    if (!evaluation) return;
    try {
      await navigator.clipboard.writeText(evaluation.prompt);
      setCopied(true);
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current);
      }
      timerRef.current = window.setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  };

  const tokenCount = evaluation ? countTokens(evaluation.prompt) : 0;
  const tokenRatio = Math.min(100, (tokenCount / Math.max(1, maxTokens)) * 100);
  const isOverflow = tokenCount > maxTokens;

  const parsed =
    evaluation && strategy === 'neurosymbolic'
      ? parseNeurosymbolicPrompt(evaluation.prompt)
      : null;

  const hasPenalty = evaluation !== null && evaluation.structural_penalty > 0;

  return (
    <section className="card" aria-label="Разбор лучшего решения">
      <div className="card-header">
        <div>
          <h2 className="card-title">Разбор лучшего решения</h2>
          <p className="card-subtitle">
            Синтезированный промпт LLM, контроль лимита токенов CLIP и декомпозиция F(X)
          </p>
        </div>

        <button
          type="button"
          className="btn-tonal"
          onClick={handleCopy}
          disabled={!evaluation || isLoading}
        >
          {copied ? <Check size={16} /> : <Copy size={16} />}
          <span>{copied ? 'Скопировано' : 'Скопировать промпт'}</span>
        </button>
      </div>

      {!evaluation ? (
        <div className="prompt-preview">
          <p className="prompt-preview__plain" style={{ color: 'var(--text-secondary)' }}>
            {isLoading
              ? 'Подбор оптимальных дескрипторов и гиперпараметров...'
              : 'Здесь появится найденный промпт с разметкой ядра, весами слотов и раскладкой метрик.'}
          </p>
        </div>
      ) : (
        <div className="solution-grid">
          <div>
            <div className="prompt-preview">
              {parsed && parsed.slots.length > 0 ? (
                <>
                  {parsed.core && (
                    <span className="prompt-tag prompt-tag--core mono">
                      {parsed.core}
                    </span>
                  )}
                  {parsed.slots.map((slot, idx) => (
                    <span
                      key={`${slot.text}-${idx}`}
                      className="prompt-tag prompt-tag--slot mono"
                    >
                      <span>{slot.text}</span>
                      <span className="prompt-tag__weight">{slot.weight}</span>
                    </span>
                  ))}
                </>
              ) : (
                <p className="prompt-preview__plain mono">{evaluation.prompt}</p>
              )}
            </div>

            <div className="token-meta">
              <div className="token-meta__row">
                <span className="token-meta__label mono">
                  Длина промпта: {tokenCount} / {maxTokens} токенов CLIP
                </span>
                <span
                  className="pill-badge mono"
                  style={
                    hasPenalty
                      ? {
                          backgroundColor: 'var(--status-error-soft)',
                          color: 'var(--status-error)',
                        }
                      : {
                          backgroundColor: 'var(--status-success-soft)',
                          color: 'var(--status-success)',
                        }
                  }
                >
                  {hasPenalty ? (
                    <AlertTriangle size={14} />
                  ) : (
                    <CheckCircle2 size={14} />
                  )}
                  <span>
                    P_struct = {evaluation.structural_penalty.toFixed(4)}
                  </span>
                </span>
              </div>

              <div className="meter-track">
                <div
                  className={`meter-fill ${
                    isOverflow ? 'meter-fill--error' : 'meter-fill--blue'
                  }`}
                  style={{ width: `${tokenRatio}%` }}
                />
              </div>
            </div>

            <div className="param-pills">
              <span className="param-pill mono">
                <Sliders size={14} />
                <span>Guidance Scale (g):</span>
                <strong>{evaluation.guidance_scale.toFixed(2)}</strong>
              </span>
              <span className="param-pill mono">
                <span>Шаги (s):</span>
                <strong>{evaluation.steps}</strong>
              </span>
              <span className="param-pill mono">
                <span>LPIPS:</span>
                <strong>{evaluation.lpips_score.toFixed(4)}</strong>
              </span>
              <span className="param-pill mono">
                <span>Время шага:</span>
                <strong>{evaluation.eval_time_sec.toFixed(2)} с</strong>
              </span>
            </div>
          </div>

          <div className="meter-stack">
            <div className="meter-item">
              <div className="meter-item__head">
                <span className="meter-item__label">
                  CLIPScore (семантическое соответствие)
                </span>
                <span className="meter-item__value mono">
                  {evaluation.clip_score.toFixed(4)}
                </span>
              </div>
              <div className="meter-track">
                <div
                  className="meter-fill meter-fill--blue"
                  style={{ width: `${Math.min(100, evaluation.clip_score * 100)}%` }}
                />
              </div>
            </div>

            <div className="meter-item">
              <div className="meter-item__head">
                <span className="meter-item__label">
                  ImageReward (визуальная эстетика)
                </span>
                <span className="meter-item__value mono">
                  {evaluation.vis_score.toFixed(4)}
                </span>
              </div>
              <div className="meter-track">
                <div
                  className="meter-fill meter-fill--green"
                  style={{ width: `${Math.min(100, evaluation.vis_score * 100)}%` }}
                />
              </div>
            </div>

            <div className="meter-item">
              <div className="meter-item__head">
                <span className="meter-item__label">
                  Интегральная оценка F(X)
                </span>
                <span className="meter-item__value mono">
                  {evaluation.fitness.toFixed(4)}
                </span>
              </div>
              <div className="meter-track">
                <div
                  className="meter-fill meter-fill--blue"
                  style={{
                    width: `${Math.min(100, evaluation.fitness * 100)}%`,
                    background: 'var(--accent-gradient)',
                  }}
                />
              </div>
            </div>

            <div className="meter-item">
              <div className="meter-item__head">
                <span className="meter-item__label">
                  Структурный штраф P_struct
                </span>
                <span className="meter-item__value mono">
                  {evaluation.structural_penalty.toFixed(4)}
                </span>
              </div>
              <div className="meter-track">
                <div
                  className={`meter-fill ${
                    hasPenalty ? 'meter-fill--error' : 'meter-fill--green'
                  }`}
                  style={{
                    width: `${Math.min(100, evaluation.structural_penalty * 100)}%`,
                  }}
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
