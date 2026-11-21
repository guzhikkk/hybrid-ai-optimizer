import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';
import { CompareView } from './components/CompareView';
import { ConvergenceChart } from './components/ConvergenceChart';
import type { ChartSeries } from './components/ConvergenceChart';
import { Header } from './components/Header';
import { KpiStrip } from './components/KpiStrip';
import { ModelInfoModal } from './components/ModelInfoModal';
import { PromptBar } from './components/PromptBar';
import { SolutionCard } from './components/SolutionCard';
import { SwarmCanvas } from './components/SwarmCanvas';
import {
  ApiError,
  DEFAULT_CONFIG,
  checkHealth,
  runCompare,
  runOptimize,
} from './services/apiClient';
import type {
  ActiveTab,
  ComparisonResult,
  ConnectionStatus,
  ExperimentConfig,
  ExperimentResult,
  ThemeMode,
} from './types/api';


const THEME_STORAGE_KEY = 'convergelab-theme';


function getInitialTheme(): ThemeMode {
  try {
    const saved = window.localStorage.getItem(THEME_STORAGE_KEY);
    if (saved === 'light' || saved === 'dark') {
      return saved;
    }
  } catch {
    // Ignore storage access errors
  }
  return 'dark';
}


export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('optimize');
  const [theme, setTheme] = useState<ThemeMode>(getInitialTheme);
  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>('checking');
  const [infoOpen, setInfoOpen] = useState(false);


  const [config, setConfig] = useState<ExperimentConfig>(DEFAULT_CONFIG);
  const [compareRuns, setCompareRuns] = useState<number>(3);


  const [optResult, setOptResult] = useState<ExperimentResult | null>(null);
  const [cmpResult, setCmpResult] = useState<ComparisonResult | null>(null);


  const [isOptimizing, setIsOptimizing] = useState(false);
  const [isComparing, setIsComparing] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);


  // Playback state for SwarmCanvas
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);


  const didAutoRunRef = useRef(false);


  // Apply theme attribute to root html element
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch {
      // Ignore storage access errors
    }
  }, [theme]);


  const handleConfigChange = useCallback((patch: Partial<ExperimentConfig>) => {
    setConfig((prev) => ({ ...prev, ...patch }));
  }, []);


  const executeOptimize = useCallback(async (targetConfig: ExperimentConfig) => {
    setIsOptimizing(true);
    setErrorMsg(null);
    setIsPlaying(false);


    try {
      const data = await runOptimize(targetConfig);
      setOptResult(data);
      setConnectionStatus('live');
      setCurrentIndex(0);
      setIsPlaying(data.history.length > 1);
    } catch (err: unknown) {
      const msg =
        err instanceof ApiError
          ? err.message
          : 'Не удалось выполнить оптимизацию.';
      setErrorMsg(msg);
      if (err instanceof ApiError && err.status === 0) {
        setConnectionStatus('offline');
      }
    } finally {
      setIsOptimizing(false);
    }
  }, []);


  const executeCompare = useCallback(
    async (targetConfig: ExperimentConfig, runs: number) => {
      setIsComparing(true);
      setErrorMsg(null);


      try {
        const data = await runCompare({
          core_prompt: targetConfig.core_prompt,
          num_particles: targetConfig.num_particles,
          max_iterations: targetConfig.max_iterations,
          target_fitness: targetConfig.target_fitness,
          seed: targetConfig.seed,
          num_runs: runs,
        });
        setCmpResult(data);
        setConnectionStatus('live');
      } catch (err: unknown) {
        const msg =
          err instanceof ApiError
            ? err.message
            : 'Не удалось выполнить сравнительный бенчмарк.';
        setErrorMsg(msg);
        if (err instanceof ApiError && err.status === 0) {
          setConnectionStatus('offline');
        }
      } finally {
        setIsComparing(false);
      }
    },
    [],
  );


  // Initial health check + automatic first run when API is live
  useEffect(() => {
    let cancelled = false;


    async function bootstrap() {
      try {
        await checkHealth();
        if (cancelled) return;
        setConnectionStatus('live');


        if (!didAutoRunRef.current) {
          didAutoRunRef.current = true;
          await executeOptimize(DEFAULT_CONFIG);
        }
      } catch {
        if (!cancelled) {
          setConnectionStatus('offline');
        }
      }
    }


    void bootstrap();
    return () => {
      cancelled = true;
    };
  }, [executeOptimize]);


  // Playback timer for SwarmCanvas
  useEffect(() => {
    const historyLen = optResult?.history.length ?? 0;
    if (!isPlaying || historyLen <= 1) return;


    const intervalMs = Math.max(120, Math.round(680 / playbackSpeed));
    const timer = window.setInterval(() => {
      setCurrentIndex((prev) => {
        if (prev >= historyLen - 1) {
          setIsPlaying(false);
          return historyLen - 1;
        }
        return prev + 1;
      });
    }, intervalMs);


    return () => window.clearInterval(timer);
  }, [isPlaying, playbackSpeed, optResult]);


  const handleTogglePlay = () => {
    if (!optResult || optResult.history.length <= 1) return;
    if (!isPlaying && currentIndex >= optResult.history.length - 1) {
      setCurrentIndex(0);
      setIsPlaying(true);
      return;
    }
    setIsPlaying((prev) => !prev);
  };


  const handleResetPlay = () => {
    setIsPlaying(false);
    setCurrentIndex(0);
  };


  const handleSelectIteration = (index: number) => {
    setIsPlaying(false);
    setCurrentIndex(index);
  };


  const history = optResult?.history ?? [];
  const safeIndex =
    history.length > 0 ? Math.min(currentIndex, history.length - 1) : 0;
  const activeLog = history.length > 0 ? history[safeIndex] : null;


  const singleChartSeries: ChartSeries[] = useMemo(() => {
    if (!optResult || optResult.history.length === 0) return [];
    return [
      {
        id: 'global-best',
        label: 'Рекорд F_best',
        color: 'var(--series-pso)',
        areaFill: true,
        values: optResult.history.map((log) => ({
          iteration: log.iteration,
          value: log.global_best_fitness,
        })),
      },
      {
        id: 'mean-fitness',
        label: 'Среднее по рою',
        color: 'var(--text-secondary)',
        dashed: true,
        values: optResult.history.map((log) => ({
          iteration: log.iteration,
          value: log.mean_fitness,
        })),
      },
    ];
  }, [optResult]);


  return (
    <div className="app-shell">
      <Header
        activeTab={activeTab}
        onTabChange={setActiveTab}
        theme={theme}
        onToggleTheme={() =>
          setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'))
        }
        connectionStatus={connectionStatus}
        onOpenInfo={() => setInfoOpen(true)}
      />


      <main className="app-main">
        {errorMsg && (
          <div className="error-banner" role="alert">
            <div className="error-banner__content">
              <AlertCircle size={20} style={{ flexShrink: 0, marginTop: '2px' }} />
              <div className="error-banner__text">
                <p className="error-banner__title">{errorMsg}</p>
                {connectionStatus === 'offline' && (
                  <p className="error-banner__hint">
                    Убедитесь, что бэкенд запущен локально (<code className="mono">python app.py</code> в папке <code className="mono">backend</code>) или на Render.
                  </p>
                )}
              </div>
            </div>


            <button
              type="button"
              className="btn-tonal"
              onClick={() => {
                if (activeTab === 'optimize') {
                  void executeOptimize(config);
                } else {
                  void executeCompare(config, compareRuns);
                }
              }}
            >
              <RefreshCw size={16} />
              <span>Повторить</span>
            </button>
          </div>
        )}


        {activeTab === 'optimize' ? (
          <>
            <PromptBar
              config={config}
              onChange={handleConfigChange}
              onRun={() => void executeOptimize(config)}
              isLoading={isOptimizing}
            />


            <KpiStrip
              metrics={optResult?.macro_metrics ?? null}
              currentLog={activeLog}
              targetFitness={optResult?.config.target_fitness ?? config.target_fitness}
              scheduler={optResult?.config.scheduler ?? config.scheduler}
              isLoading={isOptimizing}
            />


            <section className="viz-grid" aria-label="Визуализация оптимизации">
              <SwarmCanvas
                history={history}
                currentIndex={safeIndex}
                onIndexChange={handleSelectIteration}
                isPlaying={isPlaying}
                onTogglePlay={handleTogglePlay}
                onResetPlay={handleResetPlay}
                playbackSpeed={playbackSpeed}
                onSpeedChange={setPlaybackSpeed}
                theme={theme}
                isLoading={isOptimizing}
              />


              <ConvergenceChart
                title="Кривая сходимости F(t)"
                subtitle="Кликните на любую итерацию, чтобы синхронизировать кадр роя"
                series={singleChartSeries}
                targetFitness={
                  optResult?.config.target_fitness ?? config.target_fitness
                }
                activeIteration={history.length > 0 ? safeIndex : null}
                onSelectIteration={handleSelectIteration}
                isLoading={isOptimizing}
              />
            </section>


            <SolutionCard
              evaluation={activeLog?.global_best_evaluation ?? null}
              maxTokens={optResult?.config.max_tokens ?? config.max_tokens}
              strategy={optResult?.config.prompt_strategy ?? config.prompt_strategy}
              isLoading={isOptimizing}
            />
          </>
        ) : (
          <CompareView
            corePrompt={config.core_prompt}
            onCorePromptChange={(prompt) =>
              handleConfigChange({ core_prompt: prompt })
            }
            numRuns={compareRuns}
            onNumRunsChange={setCompareRuns}
            maxIterations={config.max_iterations}
            onMaxIterationsChange={(iters) =>
              handleConfigChange({ max_iterations: iters })
            }
            targetFitness={config.target_fitness}
            onTargetFitnessChange={(val) =>
              handleConfigChange({ target_fitness: val })
            }
            comparison={cmpResult}
            isLoading={isComparing}
            onRunCompare={() => void executeCompare(config, compareRuns)}
          />
        )}
      </main>


      <ModelInfoModal isOpen={infoOpen} onClose={() => setInfoOpen(false)} />
    </div>
  );
}
