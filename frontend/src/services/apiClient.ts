import type {
  CompareRequestPayload,
  ComparisonResult,
  DefaultsResponse,
  ExperimentConfig,
  ExperimentResult,
  HealthResponse,
  OptimizeRequestPayload,
} from '../types/api';

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/+$/, '');

export const DEFAULT_CONFIG: ExperimentConfig = {
  core_prompt: '3D mascot raccoon wearing a green hoodie and backpack',
  optimizer: 'pso',
  prompt_strategy: 'neurosymbolic',
  scheduler: 'ddim',
  backend_mode: 'surrogate',
  num_particles: 8,
  max_iterations: 15,
  target_fitness: 0.88,
  max_tokens: 77,
  seed: 42,
  pso_w: 0.72,
  pso_c1: 1.49,
  pso_c2: 1.49,
  ga_elitism: 2,
  ga_crossover_rate: 0.8,
  ga_mutation_rate: 0.15,
  metric_weights: {
    w_clip: 0.55,
    w_vis: 0.45,
    w_penalty: 0.5,
    alpha: 0.6,
    beta: 0.4,
  },
  slot_dictionaries: {
    style: [
      '3d Pixar style render',
      'Unreal Engine 5 stylized character',
      'isometric clay figurine',
      'Blender Cycles studio art',
    ],
    lighting: [
      'cinematic volumetric lighting',
      'soft studio rim light',
      'warm golden hour glow',
      'neon cyberpunk backlight',
    ],
    detail: [
      '8k resolution, sharp focus, intricate textures',
      'octane render, smooth shading, clean geometry',
      'ultra detailed fur and fabric, global illumination',
      'minimalist polished surface, crisp edges',
    ],
  },
};

export const EXAMPLE_PROMPTS: ReadonlyArray<{ label: string; prompt: string }> = [
  {
    label: '3D маскот-енот',
    prompt: '3D mascot raccoon wearing a green hoodie and backpack',
  },
  {
    label: 'Болид в неоновом дожде',
    prompt: 'futuristic racing car speeding through neon rain city track',
  },
  {
    label: 'Архитектурный павильон',
    prompt: 'minimalist glass pavilion over calm lake at golden hour',
  },
];

export class ApiError extends Error {
  public readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function requestJson<T>(
  path: string,
  options: RequestInit = {},
  timeoutMs = 30000,
): Promise<T> {
  const controller = new AbortController();
  const timerId = window.setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(`${API_BASE}${path}`, {
      ...options,
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json',
        ...(options.headers ?? {}),
      },
    });

    const text = await response.text();
    const data = text ? (JSON.parse(text) as Record<string, unknown>) : {};

    if (!response.ok) {
      const msg =
        typeof data.error === 'string'
          ? data.error
          : `Ошибка сервера (HTTP ${response.status})`;
      throw new ApiError(msg, response.status);
    }

    return data as T;
  } catch (err: unknown) {
    if (err instanceof ApiError) {
      throw err;
    }
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new ApiError('Превышено время ожидания ответа сервера.', 408);
    }
    throw new ApiError('Не удалось подключиться к серверу API.', 0);
  } finally {
    window.clearTimeout(timerId);
  }
}

export async function checkHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>('/api/health', { method: 'GET' }, 5000);
}

export async function fetchDefaults(): Promise<DefaultsResponse> {
  return requestJson<DefaultsResponse>('/api/config/defaults', { method: 'GET' }, 8000);
}

export async function runOptimize(
  payload: OptimizeRequestPayload,
): Promise<ExperimentResult> {
  return requestJson<ExperimentResult>(
    '/api/optimize',
    {
      method: 'POST',
      body: JSON.stringify(payload),
    },
    45000,
  );
}

export async function runCompare(
  payload: CompareRequestPayload,
): Promise<ComparisonResult> {
  return requestJson<ComparisonResult>(
    '/api/compare',
    {
      method: 'POST',
      body: JSON.stringify(payload),
    },
    60000,
  );
}