export type OptimizerType = 'random' | 'ga' | 'pso';
export type PromptStrategyType = 'zero_shot' | 'in_context' | 'neurosymbolic';
export type SchedulerType = 'ddpm' | 'ddim';
export type BackendMode = 'surrogate' | 'api' | 'colab_local';

export type ActiveTab = 'optimize' | 'compare';
export type ThemeMode = 'dark' | 'light';
export type ConnectionStatus = 'checking' | 'live' | 'offline';

export interface MetricWeights {
  w_clip: number;
  w_vis: number;
  w_penalty: number;
  alpha: number;
  beta: number;
}

export interface ExperimentConfig {
  core_prompt: string;
  optimizer: OptimizerType;
  prompt_strategy: PromptStrategyType;
  scheduler: SchedulerType;
  backend_mode: BackendMode;
  num_particles: number;
  max_iterations: number;
  target_fitness: number;
  max_tokens: number;
  seed: number | null;
  pso_w: number;
  pso_c1: number;
  pso_c2: number;
  ga_elitism: number;
  ga_crossover_rate: number;
  ga_mutation_rate: number;
  metric_weights: MetricWeights;
  slot_dictionaries: Record<string, string[]>;
}

export type OptimizeRequestPayload = Partial<ExperimentConfig>;

export interface CompareRequestPayload extends Partial<ExperimentConfig> {
  num_runs?: number;
}

export interface EvaluationBreakdown {
  prompt: string;
  guidance_scale: number;
  steps: number;
  clip_score: number;
  vis_score: number;
  structural_penalty: number;
  lpips_score: number;
  fitness: number;
  cost: number;
  eval_time_sec: number;
  image_url: string | null;
}

export interface ParticleSnapshot {
  particle_id: number;
  position: number[];
  velocity: number[];
  best_position: number[];
  best_fitness: number;
  current_evaluation: EvaluationBreakdown;
}

export interface IterationLog {
  iteration: number;
  global_best_fitness: number;
  global_best_cost: number;
  global_best_position: number[];
  global_best_evaluation: EvaluationBreakdown;
  mean_fitness: number;
  iteration_time_sec: number;
  particles: ParticleSnapshot[];
}

export interface MacroMetrics {
  baseline_fitness: number;
  final_fitness: number;
  relative_gain_pct: number;
  convergence_iteration: number | null;
  total_time_sec: number;
  fitness_std: number;
  mean_lpips: number;
}

export interface ExperimentResult {
  experiment_id: string;
  config: ExperimentConfig;
  macro_metrics: MacroMetrics;
  history: IterationLog[];
}

export interface ComparisonPresetInfo {
  preset_id: string;
  label: string;
  optimizer: OptimizerType;
  prompt_strategy: PromptStrategyType;
  scheduler: SchedulerType;
}

export interface ComparisonAggregatedStats {
  mean_final_fitness: number;
  fitness_std: number;
  mean_relative_gain_pct: number;
  mean_total_time_sec: number;
  mean_lpips: number;
  convergence_rate_pct: number;
  mean_convergence_iteration: number | null;
}

export interface ConvergencePoint {
  iteration: number;
  global_best_fitness: number;
  mean_fitness: number;
}

export interface ComparisonPresetEntry extends ComparisonPresetInfo {
  aggregated: ComparisonAggregatedStats;
  convergence_curve: ConvergencePoint[];
  representative_run: ExperimentResult;
}

export interface ComparisonSummaryRow {
  preset_id: string;
  label: string;
  mean_final_fitness: number;
  mean_relative_gain_pct: number;
  mean_convergence_iteration: number | null;
  mean_total_time_sec: number;
  fitness_std: number;
  mean_lpips: number;
}

export interface ComparisonResult {
  comparison_id: string;
  core_prompt: string;
  num_runs: number;
  target_fitness: number;
  winner_preset_id: string;
  winner_label: string;
  summary_table: ComparisonSummaryRow[];
  presets: ComparisonPresetEntry[];
}

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
}

export interface DefaultsResponse {
  defaults: ExperimentConfig;
  options: {
    optimizers: OptimizerType[];
    prompt_strategies: PromptStrategyType[];
    schedulers: SchedulerType[];
    backend_modes: BackendMode[];
  };
  comparison_presets: ComparisonPresetInfo[];
}