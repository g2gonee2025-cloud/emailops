/**
 * API Client for EmailOps Cortex Backend
 *
 * Provides typed methods for all backend endpoints.
 */

import { z } from 'zod';
import { GeneratedDraftSchema } from '../schemas/draft';
import { logger } from './logger';
import { doctorReportSchema, statusDataSchema, type DoctorReport, type StatusData } from '../schemas/admin';
import { ensureValidToken, tokenStore } from './oidc';

export type { DoctorReport, StatusData };

// =============================================================================
// Retry Configuration
// =============================================================================

/** Status codes that should trigger a retry */
const DEFAULT_RETRY_STATUS_CODES = [429, 500, 502, 503, 504] as const;

/** Configuration for retry behavior */
export interface RetryConfig {
  /** Maximum number of retry attempts (default: 3) */
  retries: number;
  /** Base delay in milliseconds for exponential backoff (default: 1000) */
  baseDelay: number;
  /** Status codes that should trigger a retry */
  retryOn: readonly number[];
}

/** Default retry configuration */
const DEFAULT_RETRY_CONFIG: RetryConfig = {
  retries: 3,
  baseDelay: 1000,
  retryOn: DEFAULT_RETRY_STATUS_CODES,
};

/** Options for the request function */
export interface RequestOptions extends RequestInit {
  /** Retry configuration. Set to false to disable retries, or provide custom config */
  retry?: boolean | Partial<RetryConfig>;
}

/**
 * Sleep for a specified duration, respecting AbortSignal
 * @param ms - Duration to sleep in milliseconds
 * @param signal - Optional AbortSignal to cancel the sleep
 */
async function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(resolve, ms);
    signal?.addEventListener('abort', () => {
      clearTimeout(timeout);
      reject(signal.reason);
    });
  });
}

/**
 * Calculate delay with jitter (±20%)
 * @param baseDelay - Base delay in milliseconds
 * @param attempt - Current attempt number (0-indexed)
 * @returns Delay with exponential backoff and jitter
 */
function calculateDelayWithJitter(baseDelay: number, attempt: number): number {
  const exponentialDelay = baseDelay * Math.pow(2, attempt);
  const jitterFactor = 0.8 + Math.random() * 0.4; // ±20% jitter
  return Math.round(exponentialDelay * jitterFactor);
}

/**
 * Parse Retry-After header value
 * @param retryAfter - Value of Retry-After header
 * @returns Delay in milliseconds, or null if invalid
 */
function parseRetryAfter(retryAfter: string | null): number | null {
  if (!retryAfter) return null;

  // Try parsing as seconds (integer)
  const seconds = parseInt(retryAfter, 10);
  if (!isNaN(seconds) && seconds >= 0) {
    return seconds * 1000;
  }

  // Try parsing as HTTP date
  const date = Date.parse(retryAfter);
  if (!isNaN(date)) {
    const delayMs = date - Date.now();
    return delayMs > 0 ? delayMs : null;
  }

  return null;
}

/**
 * Check if a status code should trigger a retry
 */
function shouldRetry(status: number, retryOn: readonly number[]): boolean {
  return retryOn.includes(status);
}

/**
 * Get resolved retry config from options
 */
function getRetryConfig(retry: boolean | Partial<RetryConfig> | undefined): RetryConfig | null {
  if (retry === false) return null;
  if (retry === true || retry === undefined) return DEFAULT_RETRY_CONFIG;
  return {
    ...DEFAULT_RETRY_CONFIG,
    ...retry,
  };
}

// =============================================================================
// Type Definitions & Zod Schemas
// =============================================================================
export interface SearchResult {
  chunk_id: string;
  conversation_id: string;
  thread_id?: string;
  content: string;
  score: number;
  metadata?: Record<string, unknown>;
}

export interface SearchResponse {
  correlation_id?: string;
  results: SearchResult[];
  total_count: number;
  query_time_ms: number;
}

export interface Answer {
  text: string;
  confidence_overall: number;
  sources?: { chunk_id: string; content: string; score: number }[];
}

export interface AnswerResponse {
  correlation_id?: string;
  answer: Answer;
  confidence: number;
  debug_info?: Record<string, unknown>;
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface ChatResponse {
  correlation_id?: string;
  action: 'answer' | 'search' | 'summarize';
  reply: string;
  answer?: Answer;
  search_results?: SearchResult[];
  debug_info?: Record<string, unknown>;
}

export interface S3Folder {
  folder: string;
  size_bytes?: number;
  last_modified?: string;
}

export interface ListS3FoldersResponse {
  prefix: string;
  folders: S3Folder[];
  count: number;
}

export interface IngestJobResponse {
  job_id: string;
  status: string;
  folders_found?: number;
  folders_to_process?: string[];
  message: string;
}

export interface IngestStatusResponse {
  job_id: string;
  status: string;
  folders_processed: number;
  threads_created: number;
  chunks_created: number;
  embeddings_generated: number;
  errors: number;
  skipped: number;
  message: string;
}

export interface SystemConfig {
  environment: string;
  provider: string;
  log_level: string;
  database_url?: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  environment: string;
}
// Draft Email Types
const DraftEmailResponseSchema = z.object({
  correlation_id: z.string().optional(),
  draft: GeneratedDraftSchema,
  confidence: z.number(),
  iterations: z.number(),
});
export type DraftEmailResponse = z.infer<typeof DraftEmailResponseSchema>;

// Summarize Types
export interface ThreadSummary {
  summary: string;
  key_points?: string[];
  action_items?: string[];
}

export interface SummarizeResponse {
  correlation_id?: string;
  summary: ThreadSummary;
}

// Push Ingest Types
export interface PushDocument {
  document_id?: string;
  source_type?: string;
  text: string;
  metadata?: Record<string, unknown>;
}

export interface PushIngestResponse {
  job_id: string;
  documents_received: number;
  documents_ingested: number;
  chunks_created: number;
  embeddings_generated: number;
  errors: string[];
  message: string;
}

// Auth Types
export interface LoginResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

// Thread Types
export interface Message {
  message_id: string;
  sender: string;
  content: string;
  timestamp: string;
}

export interface Thread {
  thread_id: string;
  subject: string;
  participants: string[];
  messages: Message[];
}

// Thread List Types (for conversation selector)
export interface ThreadListItem {
  conversation_id: string;
  subject: string | null;
  smart_subject: string | null;
  folder_name: string;
  participants_preview: string | null;
  latest_date: string | null;
}

export interface ThreadListResponse {
  threads: ThreadListItem[];
  total_count: number;
  has_more: boolean;
}

// =============================================================================
// API Client
// =============================================================================

// API Base URL - uses environment variable or defaults to relative path for Vite proxy
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

// Custom Error for API responses
export class ApiError extends Error {
  status: number;
  details?: Record<string, unknown>;

  constructor(message: string, status: number, details?: Record<string, unknown>) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.details = details;
  }
}

const getHeaders = (token?: string | null): HeadersInit => {
  const headers: HeadersInit = { 'Content-Type': 'application/json' };
  const authToken = token ?? localStorage.getItem('auth_token');
  if (authToken) {
    headers['Authorization'] = `Bearer ${authToken}`;
  }
  return headers;
};

export const request = async <T>(
  endpoint: string,
  options: RequestOptions = {},
  includeAuth = true,
): Promise<T> => {
  const { retry, ...fetchOptions } = options;
  const retryConfig = getRetryConfig(retry);

  // Ensure valid token for authenticated requests (except refresh endpoint)
  let authToken: string | null = null;
  if (includeAuth && !endpoint.includes('/auth/refresh')) {
    authToken = await ensureValidToken();
    if (!authToken && tokenStore.getRefreshToken()) {
      logger.warn('Token refresh failed, proceeding without valid token');
    }
  }

  const baseHeaders = includeAuth ? getHeaders(authToken) : {};

  const config: RequestInit = {
    ...fetchOptions,
    headers: {
      ...baseHeaders,
      ...fetchOptions.headers,
    },
  };

  const url = endpoint.startsWith('http') ? endpoint : `${API_BASE_URL}${endpoint}`;
  const signal = config.signal as AbortSignal | undefined;

  let lastError: ApiError | Error | null = null;
  const maxAttempts = retryConfig ? retryConfig.retries + 1 : 1;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      // Check if aborted before making request
      if (signal?.aborted) {
        throw signal.reason || new DOMException('Aborted', 'AbortError');
      }

      const response = await fetch(url, config);

      if (!response.ok) {
        let errorDetails;
        try {
          errorDetails = await response.json();
        } catch (_error) {
          logger.warn('Could not parse JSON error response', {
            status: response.status,
            statusText: response.statusText,
          });
          errorDetails = { detail: response.statusText };
        }

        const apiError = new ApiError(
          `API request failed: ${response.status}`,
          response.status,
          errorDetails,
        );

        // Check if we should retry this status code
        const canRetry =
          retryConfig &&
          attempt < retryConfig.retries &&
          shouldRetry(response.status, retryConfig.retryOn);

        if (canRetry) {
          lastError = apiError;

          // Calculate delay: use Retry-After header for 429, otherwise exponential backoff
          let delayMs: number;
          if (response.status === 429) {
            const retryAfterMs = parseRetryAfter(response.headers.get('Retry-After'));
            delayMs = retryAfterMs ?? calculateDelayWithJitter(retryConfig.baseDelay, attempt);
          } else {
            delayMs = calculateDelayWithJitter(retryConfig.baseDelay, attempt);
          }

          // Log retry attempt in development
          if (import.meta.env.DEV) {
            logger.warn(`Retrying request (attempt ${attempt + 1}/${retryConfig.retries})`, {
              endpoint,
              status: response.status,
              delayMs,
            });
          }

          await sleep(delayMs, signal);
          continue;
        }

        // Non-retryable error or max retries reached
        if (response.status === 401 && !response.url.includes('login')) {
          globalThis.dispatchEvent(new CustomEvent('cortex-unauthorized'));
        } else if (response.status !== 401) {
          globalThis.dispatchEvent(new CustomEvent('api:error', { detail: apiError }));
        }

        throw apiError;
      }

      if (response.status === 204) {
        return {} as T;
      }

      return (await response.json()) as T;
    } catch (error) {
      // Re-throw abort errors immediately without retry
      if (error instanceof DOMException && error.name === 'AbortError') {
        throw error;
      }

      // Re-throw ApiError if it's non-retryable or we've exhausted retries
      if (error instanceof ApiError) {
        throw error;
      }

      // Network errors - check if we should retry
      const canRetryNetwork = retryConfig && attempt < retryConfig.retries;

      if (canRetryNetwork) {
        lastError = error instanceof Error ? error : new Error(String(error));

        const delayMs = calculateDelayWithJitter(retryConfig.baseDelay, attempt);

        if (import.meta.env.DEV) {
          logger.warn(`Retrying request after network error (attempt ${attempt + 1}/${retryConfig.retries})`, {
            endpoint,
            error: error instanceof Error ? error.message : String(error),
            delayMs,
          });
        }

        await sleep(delayMs, signal);
        continue;
      }

      // Max retries exhausted for network error
      logger.error('Network or other fetch error occurred.', {
        endpoint: endpoint,
        error: error instanceof Error ? error.message : String(error),
        attempts: attempt + 1,
      });
      throw new Error('A network error occurred.');
    }
  }

  // This should not be reached, but handle it just in case
  if (lastError) {
    throw lastError;
  }
  throw new Error('Request failed after all retry attempts');
};

export const api = {
  // ---------------------------------------------------------------------------
  // System Endpoints
  // ---------------------------------------------------------------------------

  fetchHealth: (signal?: AbortSignal): Promise<HealthResponse> =>
    request<HealthResponse>('/health', { signal }, false),

  fetchVersion: (signal?: AbortSignal): Promise<Record<string, string>> =>
    request<Record<string, string>>('/version', { signal }, false),

  // ---------------------------------------------------------------------------
  // RAG Endpoints
  // ---------------------------------------------------------------------------

  search: (
    query: string,
    k = 10,
    filters: Record<string, unknown> = {},
    signal?: AbortSignal,
  ): Promise<SearchResponse> => {
    return request<SearchResponse>('/api/v1/search', {
      method: 'POST',
      body: JSON.stringify({ query, k, filters }),
      signal,
    });
  },

  ask: (
    query: string,
    threadId?: string,
    k = 10,
    signal?: AbortSignal,
  ): Promise<AnswerResponse> => {
    return request<AnswerResponse>('/api/v1/answer', {
      method: 'POST',
      body: JSON.stringify({ query, thread_id: threadId, k }),
      signal,
    });
  },

  chat: (
    messages: ChatMessage[],
    threadId?: string,
    k = 10,
    signal?: AbortSignal,
  ): Promise<ChatResponse> => {
    return request<ChatResponse>('/api/v1/chat', {
      method: 'POST',
      body: JSON.stringify({ messages, thread_id: threadId, k }),
      signal,
    });
  },

  // ---------------------------------------------------------------------------
  // Ingestion Endpoints
  // ---------------------------------------------------------------------------

  listS3Folders: (
    prefix = 'Outlook/',
    limit = 100,
    signal?: AbortSignal,
  ): Promise<ListS3FoldersResponse> => {
    const params = new URLSearchParams({ prefix, limit: String(limit) });
    return request<ListS3FoldersResponse>(`/api/v1/ingest/list?${params}`, { signal });
  },

  startIngestion: (
    prefix = 'Outlook/',
    limit?: number,
    dryRun = false,
    signal?: AbortSignal,
  ): Promise<IngestJobResponse> => {
    return request<IngestJobResponse>('/api/v1/ingest/s3', {
      method: 'POST',
      body: JSON.stringify({ prefix, limit, dry_run: dryRun }),
      signal,
    });
  },

  getIngestionStatus: (jobId: string, signal?: AbortSignal): Promise<IngestStatusResponse> => {
    return request<IngestStatusResponse>(`/api/v1/ingest/status/${jobId}`, { signal });
  },

  // ---------------------------------------------------------------------------
  // Admin Endpoints
  // ---------------------------------------------------------------------------

  runDoctor: async (signal?: AbortSignal): Promise<DoctorReport> => {
    const data = await request<unknown>('/api/v1/admin/doctor', { method: 'POST', signal });
    return doctorReportSchema.parse(data);
  },

  fetchStatus: async (signal?: AbortSignal): Promise<StatusData> => {
    const data = await request<unknown>('/api/v1/admin/status', { signal });
    return statusDataSchema.parse(data);
  },

  fetchConfig: (signal?: AbortSignal): Promise<SystemConfig> =>
    request<SystemConfig>('/api/v1/admin/config', { signal }),

  // ---------------------------------------------------------------------------
  // Draft & Summarize Endpoints
  // ---------------------------------------------------------------------------

  draftEmail: async (
    instruction: string,
    threadId?: string,
    tone = 'professional',
    signal?: AbortSignal,
  ): Promise<DraftEmailResponse> => {
    const response = await request<unknown>('/api/v1/draft-email', {
      method: 'POST',
      body: JSON.stringify({ instruction, thread_id: threadId, tone }),
      signal,
    });
    return DraftEmailResponseSchema.parse(response);
  },

  summarizeThread: (
    threadId: string,
    maxLength = 500,
    signal?: AbortSignal,
  ): Promise<SummarizeResponse> => {
    return request<SummarizeResponse>('/api/v1/summarize', {
      method: 'POST',
      body: JSON.stringify({ thread_id: threadId, max_length: maxLength }),
      signal,
    });
  },

  pushDocuments: (
    documents: PushDocument[],
    generateEmbeddings = true,
    signal?: AbortSignal,
  ): Promise<PushIngestResponse> => {
    return request<PushIngestResponse>('/api/v1/ingest/push', {
      method: 'POST',
      body: JSON.stringify({ documents, generate_embeddings: generateEmbeddings }),
      signal,
    });
  },

  fetchThread: (threadId: string, signal?: AbortSignal): Promise<Thread> => {
    return request<Thread>(`/api/v1/thread/${threadId}`, { signal });
  },

  listThreads: (
    query?: string,
    limit = 50,
    offset = 0,
    signal?: AbortSignal,
  ): Promise<ThreadListResponse> => {
    const params = new URLSearchParams();
    if (query) params.set('q', query);
    params.set('limit', String(limit));
    params.set('offset', String(offset));
    return request<ThreadListResponse>(`/api/v1/threads?${params}`, { signal });
  },

  // ---------------------------------------------------------------------------
  // Auth Endpoints
  // ---------------------------------------------------------------------------

  login: (username: string, password: string): Promise<LoginResponse> => {
    return request<LoginResponse>(
      '/api/v1/auth/login',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password }),
      },
      false, // Do not include auth token for login request
    );
  },

  setAuthToken: (token: string | null) => {
    if (token) {
      localStorage.setItem('auth_token', token);
    } else {
      localStorage.removeItem('auth_token');
    }
  },
};

export default api;
