import { useEffect } from 'react';
import { Layers, Sigma, X } from 'lucide-react';


interface ModelInfoModalProps {
  isOpen: boolean;
  onClose: () => void;
}


const ARCHITECTURE_PATTERNS = [
  {
    name: 'Strategy (Стратегия)',
    desc: 'Взаимозаменяемые алгоритмы поиска (PSO, GA, Random) и стратегии LLM (Neurosymbolic, ICL, Zero-shot).',
  },
  {
    name: 'Facade (Фасад)',
    desc: 'EvaluationFacade инкапсулирует декодирование промпта, вызов диффузии и расчёт интегрального F(X).',
  },
  {
    name: 'Adapter (Адаптер)',
    desc: 'Единый интерфейс для 3 режимов исполнения: Surrogate (демо/тесты), External API и Colab GPU.',
  },
  {
    name: 'Factory Method & Observer',
    desc: 'Сборка связок эксперимента через ExperimentFactory и независимое логирование траекторий.',
  },
] as const;


export function ModelInfoModal({ isOpen, onClose }: ModelInfoModalProps) {
  useEffect(() => {
    if (!isOpen) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [isOpen, onClose]);


  if (!isOpen) return null;


  return (
    <div
      className="modal-backdrop"
      role="presentation"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className="modal-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="model-modal-title"
      >
        <div className="modal-header">
          <div>
            <h2 id="model-modal-title" className="modal-title">
              Математическая модель ConvergeLab
            </h2>
            <p className="modal-subtitle">
              Исследовательский стенд оптимизации гибридного ИИ-конвейера «LLM + Латентная диффузия»
            </p>
          </div>


          <button
            type="button"
            className="btn-icon"
            onClick={onClose}
            aria-label="Закрыть окно"
          >
            <X size={16} />
          </button>
        </div>


        <div className="modal-section">
          <h3 className="modal-section__title">
            <Sigma size={16} />
            <span>Интегральная целевая функция и штраф</span>
          </h3>
          <div className="formula-box mono">
            <div>F(X) = 0.55 · S_CLIP + 0.45 · S_Vis − 0.50 · P_struct</div>
            <div>J(X) = 1 − F(X) → min</div>
            <div style={{ marginTop: '8px', color: 'var(--text-secondary)', fontSize: '12px' }}>
              P_struct = 0.60 · (1 − R_core) + 0.40 · max(0, (|P| − 77) / 77)
            </div>
          </div>
          <p className="card-subtitle" style={{ fontSize: '13px', lineHeight: '20px' }}>
            Каждая частица кодирует 8-мерный вектор X = [g, s, w₁, w₂, w₃, z₁, z₂, z₃]:
            Guidance Scale, число шагов денойзинга, веса внимания слотов и селекторы дескрипторов LLM.
          </p>
        </div>


        <div className="modal-section">
          <h3 className="modal-section__title">
            <Layers size={16} />
            <span>Архитектурные паттерны ядра</span>
          </h3>
          <div className="pattern-list">
            {ARCHITECTURE_PATTERNS.map((item) => (
              <div key={item.name} className="pattern-item">
                <span className="pattern-item__name">{item.name}</span>
                <span className="pattern-item__desc">{item.desc}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
