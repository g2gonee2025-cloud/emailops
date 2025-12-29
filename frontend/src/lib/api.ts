/**
 * API Client for EmailOps Cortex Backend
 *
 * Provides typed methods for all backend endpoints.
 */

import { logger } from './logger';

// =============================================================================
// Type Definitions
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

export interface DoctorCheck {
  name: string;
  status: 'pass' | 'fail' | 'warn';
  message?: string;
  details?: Record<string, unknown>;
}

export interface DoctorReport {
  overall_status: 'healthy' | 'degraded' | 'unhealthy';
  checks: DoctorCheck[];
}

export interface SystemStatus {
  status: string;
  service: string;
  env: string;
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
export interface EmailDraft {
  subject: string;
  body: string;
  to?: string[];
  cc?: string[];
}

export interface DraftEmailResponse {
  correlation_id?: string;
  draft: EmailDraft;
  confidence: number;
  iterations: number;
}

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

// =============================================================================
// API Client
// =============================================================================

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

/**
 * Creates the headers for an API request.
 * @param includeAuth - Whether to include the Authorization header. Defaults to true.
 * @returns A HeadersInit object.
 */
const getHeaders = (includeAuth = true): HeadersInit => {
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
  };

  if (includeAuth) {
    const authToken = localStorage.getItem('auth_token');
    if (authToken) {
      headers['Authorization'] = `Bearer ${authToken}`;
    }
  }

  return headers;
};

/**
 * A generic request wrapper for the API client.
 *
 * - Ensures consistent headers (including auth).
 * - Handles non-OK responses by throwing a typed ApiError.
 * - Safely parses JSON.
 * - Supports AbortController signals.
 * - Avoids logging sensitive data.
 *
 * @param endpoint - The API endpoint to call.
 * @param options - RequestInit options for fetch.
 * @param includeAuth - Whether to include the auth token.
 * @returns The JSON response as type T.
 */
export const request = async <T>(
  endpoint: string,
  options: RequestInit = {},
  includeAuth = true,
): Promise<T> => {
  const config: RequestInit = {
    ...options,
    // Merge default headers with any custom headers from options
    headers: { ...getHeaders(includeAuth), ...options.headers },
  };

  try {
    const response = await fetch(endpoint, config);

    if (!response.ok) {
      if (response.status === 401 && !response.url.includes('login')) {
        window.dispatchEvent(new CustomEvent('cortex-unauthorized'));
      }

      let errorDetails;
      try {
        errorDetails = await response.json();
      } catch (e) {
        errorDetails = { detail: response.statusText };
      }
      const apiError = new ApiError(
        `API request failed with status ${response.status}`,
        response.status,
        errorDetails,
      );
      if (response.status !== 401) {
        window.dispatchEvent(new CustomEvent('api:error', { detail: apiError }));
      }
      throw apiError;
    }

    if (response.status === 204) {
      return {} as T;
    }

    try {
      return (await response.json()) as T;
    } catch (e) {
      const parseError = new ApiError('Failed to parse JSON response.', response.status);
      window.dispatchEvent(new CustomEvent('api:error', { detail: parseError }));
      throw parseError;
    }
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }

    const sanitizedConfig = { ...config };
    // Don't log the body of urlencoded forms
    if (
      sanitizedConfig.body &&
      (sanitizedConfig.headers as Record<string, string>)['Content-Type'] ===
        'application/x-www-form-urlencoded'
    ) {
      sanitizedConfig.body = '[REDACTED FORM]';
    } else if (sanitizedConfig.body) {
      sanitizedConfig.body = '[REDACTED]';
    }

    logger.error('A network or other fetch error occurred.', {
      endpoint: endpoint,
      config: sanitizedConfig,
      error: error instanceof Error ? error.message : String(error),
    });

    const networkError = new ApiError(
      'A network error occurred. Please try again later.',
      0, // Using 0 for network errors
    );
    window.dispatchEvent(new CustomEvent('api:error', { detail: networkError }));
    throw networkError;
  }
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

  runDoctor: (signal?: AbortSignal): Promise<DoctorReport> => {
    // Note: runDoctor uses an authenticated POST, which is now supported.
    return request<DoctorReport>('/api/v1/admin/doctor', { method: 'POST', signal }, true);
  },

  fetchStatus: (signal?: AbortSignal): Promise<SystemStatus> =>
    request<SystemStatus>('/api/v1/admin/status', { signal }),

  fetchConfig: (signal?: AbortSignal): Promise<SystemConfig> =>
    request<SystemConfig>('/api/v1/admin/config', { signal }),

  // ---------------------------------------------------------------------------
  // Draft & Summarize Endpoints
  // ---------------------------------------------------------------------------

  draftEmail: (
    instruction: string,
    threadId?: string,
    tone = 'professional',
    signal?: AbortSignal,
  ): Promise<DraftEmailResponse> => {
    return request<DraftEmailResponse>('/api/v1/draft', {
      method: 'POST',
      body: JSON.stringify({ instruction, thread_id: threadId, tone }),
      signal,
    });
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

  // ---------------------------------------------------------------------------
  // Auth Endpoints
  // ---------------------------------------------------------------------------

  login: (username: string, password: string): Promise<LoginResponse> => {
    const formData = new URLSearchParams();
    formData.append('username', username);
    formData.append('password', password);

    return request<LoginResponse>(
      '/api/v1/auth/login',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: formData,
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
