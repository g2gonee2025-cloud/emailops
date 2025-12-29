/**
 * API Client for EmailOps Cortex Backend
 *
 * Provides typed methods for all backend endpoints.
 */

import { logger } from './logger';

// =============================================================================
// Type Definitions
// =============================================================================

export class ApiError extends Error {
  status: number;
  details?: Record<string, unknown>;

  constructor(status: number, message: string, details?: Record<string, unknown>) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.details = details;
  }
}

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

// Auth token storage
let authToken: string | null = null;

const getHeaders = (): HeadersInit => {
  const headers: HeadersInit = { 'Content-Type': 'application/json' };
  if (authToken) {
    headers['Authorization'] = `Bearer ${authToken}`;
  }
  return headers;
};

interface RequestOptions extends Omit<RequestInit, 'body'> {
  body?: unknown;
}

const request = async <T>(url: string, options: RequestOptions = {}): Promise<T> => {
  const { body, ...fetchOptions } = options;

  const headers = getHeaders();

  try {
    const response = await fetch(url, {
      ...fetchOptions,
      headers: {
        ...headers,
        ...fetchOptions.headers,
      },
      body: body ? JSON.stringify(body) : undefined,
      signal: fetchOptions.signal,
    });

    if (response.status === 401) {
      window.dispatchEvent(new Event('unauthorized'));
    }

    if (!response.ok) {
      let errorData: { detail?: string } | null = null;
      try {
        errorData = await response.json();
      } catch (e) {
        // Response is not JSON or empty
      }
      throw new ApiError(
        response.status,
        errorData?.detail || response.statusText || 'An unknown error occurred',
        errorData ?? undefined
      );
    }

    // Handle 204 No Content
    if (response.status === 204) {
      return undefined as T;
    }

    const responseText = await response.text();
    try {
        return JSON.parse(responseText);
    } catch (e) {
        throw new ApiError(500, "Failed to parse JSON response", { response: responseText });
    }

  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }

    // Avoid logging sensitive data from the request body
    const { body: _, ...loggedOptions } = options;
    logger.error('API request failed unexpectedly', { url, options: loggedOptions, error });

    throw new ApiError(0, (error as Error).message || 'A network error occurred');
  }
};

export const api = {
  // ---------------------------------------------------------------------------
  // System Endpoints
  // ---------------------------------------------------------------------------

  fetchHealth: (signal?: AbortSignal) => {
    return request<HealthResponse>('/health', { signal });
  },

  fetchVersion: (signal?: AbortSignal) => {
    return request<Record<string, string>>('/version', { signal });
  },

  // ---------------------------------------------------------------------------
  // RAG Endpoints
  // ---------------------------------------------------------------------------

  search: (query: string, k = 10, filters: Record<string, unknown> = {}, signal?: AbortSignal) => {
    return request<SearchResponse>('/api/v1/search', {
      method: 'POST',
      body: { query, k, filters },
      signal,
    });
  },

  ask: (query: string, threadId?: string, k = 10, signal?: AbortSignal) => {
    return request<AnswerResponse>('/api/v1/answer', {
      method: 'POST',
      body: { query, thread_id: threadId, k },
      signal,
    });
  },

  chat: (messages: ChatMessage[], threadId?: string, k = 10, signal?: AbortSignal) => {
    return request<ChatResponse>('/api/v1/chat', {
      method: 'POST',
      body: { messages, thread_id: threadId, k },
      signal,
    });
  },

  // ---------------------------------------------------------------------------
  // Ingestion Endpoints
  // ---------------------------------------------------------------------------

  listS3Folders: (prefix = 'Outlook/', limit = 100, signal?: AbortSignal) => {
    const params = new URLSearchParams({ prefix, limit: String(limit) });
    return request<ListS3FoldersResponse>(`/api/v1/ingest/list?${params}`, { signal });
  },

  startIngestion: (prefix = 'Outlook/', limit?: number, dryRun = false, signal?: AbortSignal) => {
    return request<IngestJobResponse>('/api/v1/ingest/s3', {
      method: 'POST',
      body: { prefix, limit, dry_run: dryRun },
      signal,
    });
  },

  getIngestionStatus: (jobId: string, signal?: AbortSignal) => {
    return request<IngestStatusResponse>(`/api/v1/ingest/status/${jobId}`, { signal });
  },

  // ---------------------------------------------------------------------------
  // Admin Endpoints
  // ---------------------------------------------------------------------------

  runDoctor: (signal?: AbortSignal) => {
    return request<DoctorReport>('/api/v1/admin/doctor', { method: 'POST', signal });
  },

  fetchStatus: (signal?: AbortSignal) => {
    return request<SystemStatus>('/api/v1/admin/status', { signal });
  },

  fetchConfig: (signal?: AbortSignal) => {
    return request<SystemConfig>('/api/v1/admin/config', { signal });
  },

  // ---------------------------------------------------------------------------
  // Draft & Summarize Endpoints
  // ---------------------------------------------------------------------------

  draftEmail: (instruction: string, threadId?: string, tone = 'professional', signal?: AbortSignal) => {
    return request<DraftEmailResponse>('/api/v1/draft', {
      method: 'POST',
      body: { instruction, thread_id: threadId, tone },
      signal,
    });
  },

  summarizeThread: (threadId: string, maxLength = 500, signal?: AbortSignal) => {
    return request<SummarizeResponse>('/api/v1/summarize', {
      method: 'POST',
      body: { thread_id: threadId, max_length: maxLength },
      signal,
    });
  },

  pushDocuments: (documents: PushDocument[], generateEmbeddings = true, signal?: AbortSignal) => {
    return request<PushIngestResponse>('/api/v1/ingest/push', {
      method: 'POST',
      body: { documents, generate_embeddings: generateEmbeddings },
      signal,
    });
  },

  // ---------------------------------------------------------------------------
  // Auth Endpoints
  // ---------------------------------------------------------------------------

  login: (username: string, password: string, signal?: AbortSignal) => {
    return request<LoginResponse>('/api/v1/auth/login', {
      method: 'POST',
      body: { username, password },
      // Skip auth header for login
      headers: { Authorization: '' },
      signal,
    });
  },

  setAuthToken: (token: string | null) => {
    authToken = token;
  },
};

export default api;
