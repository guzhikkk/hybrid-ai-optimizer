import {
  BarChart3,
  Info,
  Moon,
  Orbit,
  Sparkles,
  Sun,
} from 'lucide-react';
import type { ActiveTab, ConnectionStatus, ThemeMode } from '../types/api';

interface HeaderProps {
  activeTab: ActiveTab;
  onTabChange: (tab: ActiveTab) => void;
  theme: ThemeMode;
  onToggleTheme: () => void;
  connectionStatus: ConnectionStatus;
  onOpenInfo: () => void;
}

export function Header({
  activeTab,
  onTabChange,
  theme,
  onToggleTheme,
  connectionStatus,
  onOpenInfo,
}: HeaderProps) {
  const statusLabel =
    connectionStatus === 'live'
      ? 'API Live'
      : connectionStatus === 'offline'
        ? 'Offline Replay'
        : 'Проверка API';

  return (
    <header className="app-header">
      <div className="app-header__inner">
        <div className="brand">
          <div className="brand__mark" aria-hidden="true">
            <Orbit size={20} strokeWidth={2.2} />
          </div>
          <div className="brand__text">
            <span className="brand__name">ConvergeLab</span>
            <span className="brand__tag">Hybrid AI Pipeline Optimizer</span>
          </div>
        </div>

        <nav className="nav-segmented" aria-label="Режимы работы">
          <button
            type="button"
            className={`nav-segmented__btn ${
              activeTab === 'optimize' ? 'nav-segmented__btn--active' : ''
            }`}
            onClick={() => onTabChange('optimize')}
          >
            <Sparkles size={16} />
            <span>Оптимизация</span>
          </button>
          <button
            type="button"
            className={`nav-segmented__btn ${
              activeTab === 'compare' ? 'nav-segmented__btn--active' : ''
            }`}
            onClick={() => onTabChange('compare')}
          >
            <BarChart3 size={16} />
            <span>Сравнение методов</span>
          </button>
        </nav>

        <div className="header-actions">
          <div
            className={`status-pill status-pill--${connectionStatus}`}
            title="Состояние подключения к серверному ядру Python"
          >
            <span className="status-pill__dot" />
            <span>{statusLabel}</span>
          </div>

          <button
            type="button"
            className="btn-icon"
            onClick={onOpenInfo}
            aria-label="О математической модели и архитектуре"
            title="О модели и формулах"
          >
            <Info size={16} />
          </button>

          <button
            type="button"
            className="btn-icon"
            onClick={onToggleTheme}
            aria-label="Переключить тему оформления"
            title={theme === 'dark' ? 'Светлая тема' : 'Тёмная тема'}
          >
            {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
          </button>
        </div>
      </div>
    </header>
  );
}