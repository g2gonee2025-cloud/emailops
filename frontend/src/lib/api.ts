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

// Auth token storage
let authToken: string | null = null;

// Custom Error for API responses
export class ApiError extends Error {
  status: number;
  statusText: string;
  detail?: string;

  constructor(status: number, statusText: string, detail?: string) {
    super(detail || `${status} ${statusText}`);
    this.name = 'ApiError';
    this.status = status;
    this.statusText = statusText;
    this.detail = detail;
  }
}

const handleResponse = async <T>(response: Response): Promise<T> => {
  if (!response.ok) {
    if (response.status === 401 && !response.url.includes('login')) {
      // Dispatch a custom event for unauthorized access from non-login endpoints
      window.dispatchEvent(new CustomEvent('cortex-unauthorized'));
    }
    const errorBody = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new ApiError(response.status, response.statusText, errorBody.detail);
  }
  return response.json();
};

const getHeaders = (includeAuth = true): HeadersInit => {
  const headers: HeadersInit = { 'Content-Type': 'application/json' };
  if (includeAuth && authToken) {
    headers['Authorization'] = `Bearer ${authToken}`;
  }
  return headers;
};

export const request = async <T>(
  endpoint: string,
  options: RequestInit = {},
  includeAuth = true,
): Promise<T> => {
  const headers = getHeaders(includeAuth);
  const config: RequestInit = {
    ...options,
    headers: {
      ...headers,
      ...options.headers,
    },
  };

  const response = await fetch(endpoint, config);
  return handleResponse<T>(response);
};

export const api = {
  // ---------------------------------------------------------------------------
  // System Endpoints
  // ---------------------------------------------------------------------------

  fetchHealth: async (): Promise<HealthResponse | null> => {
    try {
      return await request<HealthResponse>('/health', {}, false);
    } catch (error) {
      logger.error('Failed to fetch health:', error);
      return null;
    }
  },

  fetchVersion: async (): Promise<Record<string, string> | null> => {
    try {
      return await request<Record<string, string>>('/version', {}, false);
    } catch (error) {
      logger.error('Failed to fetch version:', error);
      return null;
    }
  },

  // ---------------------------------------------------------------------------
  // RAG Endpoints
  // ---------------------------------------------------------------------------

  search: async (query: string, k = 10, filters: Record<string, unknown> = {}): Promise<SearchResponse> => {
    return await request<SearchResponse>('/api/v1/search', {
      method: 'POST',
      body: JSON.stringify({ query, k, filters }),
    });
  },

  ask: async (query: string, threadId?: string, k = 10): Promise<AnswerResponse> => {
    return await request<AnswerResponse>('/api/v1/answer', {
      method: 'POST',
      body: JSON.stringify({ query, thread_id: threadId, k }),
    });
  },

  chat: async (messages: ChatMessage[], threadId?: string, k = 10): Promise<ChatResponse> => {
    return await request<ChatResponse>('/api/v1/chat', {
      method: 'POST',
      body: JSON.stringify({ messages, thread_id: threadId, k }),
    });
  },

  // ---------------------------------------------------------------------------
  // Ingestion Endpoints
  // ---------------------------------------------------------------------------

  listS3Folders: async (prefix = 'Outlook/', limit = 100): Promise<ListS3FoldersResponse> => {
    const params = new URLSearchParams({ prefix, limit: String(limit) });
    return await request<ListS3FoldersResponse>(`/api/v1/ingest/list?${params}`);
  },

  startIngestion: async (prefix = 'Outlook/', limit?: number, dryRun = false): Promise<IngestJobResponse> => {
    return await request<IngestJobResponse>('/api/v1/ingest/s3', {
      method: 'POST',
      body: JSON.stringify({ prefix, limit, dry_run: dryRun }),
    });
  },

  getIngestionStatus: async (jobId: string): Promise<IngestStatusResponse> => {
    return await request<IngestStatusResponse>(`/api/v1/ingest/status/${jobId}`);
  },

  // ---------------------------------------------------------------------------
  // Admin Endpoints
  // ---------------------------------------------------------------------------

  runDoctor: async (): Promise<DoctorReport> => {
    return await request<DoctorReport>('/api/v1/admin/doctor', { method: 'POST' });
  },

  fetchStatus: async (): Promise<SystemStatus | null> => {
    try {
      return await request<SystemStatus>('/api/v1/admin/status');
    } catch (error) {
      logger.error('Failed to fetch status:', error);
      return null;
    }
  },

  fetchConfig: async (): Promise<SystemConfig | null> => {
    try {
      return await request<SystemConfig>('/api/v1/admin/config');
    } catch (error) {
      logger.error('Failed to fetch config:', error);
      return null;
    }
  },

  // ---------------------------------------------------------------------------
  // Draft & Summarize Endpoints
  // ---------------------------------------------------------------------------

  draftEmail: async (instruction: string, threadId?: string, tone = 'professional'): Promise<DraftEmailResponse> => {
    return await request<DraftEmailResponse>('/api/v1/draft', {
      method: 'POST',
      body: JSON.stringify({ instruction, thread_id: threadId, tone }),
    });
  },

  summarizeThread: async (threadId: string, maxLength = 500): Promise<SummarizeResponse> => {
    return await request<SummarizeResponse>('/api/v1/summarize', {
      method: 'POST',
      body: JSON.stringify({ thread_id: threadId, max_length: maxLength }),
    });
  },

  pushDocuments: async (documents: PushDocument[], generateEmbeddings = true): Promise<PushIngestResponse> => {
    return await request<PushIngestResponse>('/api/v1/ingest/push', {
      method: 'POST',
      body: JSON.stringify({ documents, generate_embeddings: generateEmbeddings }),
    });
  },

  // ---------------------------------------------------------------------------
  // Auth Endpoints
  // ---------------------------------------------------------------------------

  login: async (username: string, password: string): Promise<LoginResponse> => {
    return await request<LoginResponse>(
      '/api/v1/auth/login',
      {
        method: 'POST',
        body: JSON.stringify({ username, password }),
      },
      false, // Do not include auth token for login request
    );
  },

  setAuthToken: (token: string | null) => {
    authToken = token;
  },
};

export default api;
