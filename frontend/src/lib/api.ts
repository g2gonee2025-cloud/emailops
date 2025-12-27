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

const handleResponse = async <T>(response: Response): Promise<T> => {
  if (response.status === 401) {
    window.dispatchEvent(new Event('unauthorized'));
  }
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
  return response.json();
};

const getHeaders = (): HeadersInit => {
  const headers: HeadersInit = { 'Content-Type': 'application/json' };
  if (authToken) {
    headers['Authorization'] = `Bearer ${authToken}`;
  }
  return headers;
};

export const api = {
  // ---------------------------------------------------------------------------
  // System Endpoints
  // ---------------------------------------------------------------------------

  fetchHealth: async (): Promise<HealthResponse | null> => {
    try {
      const response = await fetch('/health');
      return handleResponse<HealthResponse>(response);
    } catch (error) {
      logger.error('Failed to fetch health:', error);
      return null;
    }
  },

  fetchVersion: async (): Promise<Record<string, string> | null> => {
    try {
      const response = await fetch('/version');
      return handleResponse<Record<string, string>>(response);
    } catch (error) {
      logger.error('Failed to fetch version:', error);
      return null;
    }
  },

  // ---------------------------------------------------------------------------
  // RAG Endpoints
  // ---------------------------------------------------------------------------

  search: async (query: string, k = 10, filters: Record<string, unknown> = {}): Promise<SearchResponse> => {
    const response = await fetch('/api/v1/search', {
      method: 'POST',
      headers: getHeaders(),
      body: JSON.stringify({ query, k, filters }),
    });
    return handleResponse<SearchResponse>(response);
  },

  ask: async (query: string, threadId?: string, k = 10): Promise<AnswerResponse> => {
    const response = await fetch('/api/v1/answer', {
      method: 'POST',
      headers: getHeaders(),
      body: JSON.stringify({ query, thread_id: threadId, k }),
    });
    return handleResponse<AnswerResponse>(response);
  },

  chat: async (messages: ChatMessage[], threadId?: string, k = 10): Promise<ChatResponse> => {
    const response = await fetch('/api/v1/chat', {
      method: 'POST',
      headers: getHeaders(),
      body: JSON.stringify({ messages, thread_id: threadId, k }),
    });
    return handleResponse<ChatResponse>(response);
  },

  // ---------------------------------------------------------------------------
  // Ingestion Endpoints
  // ---------------------------------------------------------------------------

  listS3Folders: async (prefix = 'Outlook/', limit = 100): Promise<ListS3FoldersResponse> => {
    const params = new URLSearchParams({ prefix, limit: String(limit) });
    const response = await fetch(`/api/v1/ingest/list?${params}`);
    return handleResponse<ListS3FoldersResponse>(response);
  },

  startIngestion: async (prefix = 'Outlook/', limit?: number, dryRun = false): Promise<IngestJobResponse> => {
    const response = await fetch('/api/v1/ingest/s3', {
      method: 'POST',
      headers: getHeaders(),
      body: JSON.stringify({ prefix, limit, dry_run: dryRun }),
    });
    return handleResponse<IngestJobResponse>(response);
  },

  getIngestionStatus: async (jobId: string): Promise<IngestStatusResponse> => {
    const response = await fetch(`/api/v1/ingest/status/${jobId}`);
    return handleResponse<IngestStatusResponse>(response);
  },

  // ---------------------------------------------------------------------------
  // Admin Endpoints
  // ---------------------------------------------------------------------------

  runDoctor: async (): Promise<DoctorReport> => {
    const response = await fetch('/api/v1/admin/doctor', { method: 'POST' });
    return handleResponse<DoctorReport>(response);
  },

  fetchStatus: async (): Promise<SystemStatus | null> => {
    try {
      const response = await fetch('/api/v1/admin/status');
      return handleResponse<SystemStatus>(response);
    } catch (error) {
      logger.error('Failed to fetch status:', error);
      return null;
    }
  },

  fetchConfig: async (): Promise<SystemConfig | null> => {
    try {
      const response = await fetch('/api/v1/admin/config');
      return handleResponse<SystemConfig>(response);
    } catch (error) {
      logger.error('Failed to fetch config:', error);
      return null;
    }
  },

  // ---------------------------------------------------------------------------
  // Draft & Summarize Endpoints
  // ---------------------------------------------------------------------------

  draftEmail: async (instruction: string, threadId?: string, tone = 'professional'): Promise<DraftEmailResponse> => {
    const response = await fetch('/api/v1/draft', {
      method: 'POST',
      headers: getHeaders(),
      body: JSON.stringify({ instruction, thread_id: threadId, tone }),
    });
    return handleResponse<DraftEmailResponse>(response);
  },

  summarizeThread: async (threadId: string, maxLength = 500): Promise<SummarizeResponse> => {
    const response = await fetch('/api/v1/summarize', {
      method: 'POST',
      headers: getHeaders(),
      body: JSON.stringify({ thread_id: threadId, max_length: maxLength }),
    });
    return handleResponse<SummarizeResponse>(response);
  },

  pushDocuments: async (documents: PushDocument[], generateEmbeddings = true): Promise<PushIngestResponse> => {
    const response = await fetch('/api/v1/ingest/push', {
      method: 'POST',
      headers: getHeaders(),
      body: JSON.stringify({ documents, generate_embeddings: generateEmbeddings }),
    });
    return handleResponse<PushIngestResponse>(response);
  },

  // ---------------------------------------------------------------------------
  // Auth Endpoints
  // ---------------------------------------------------------------------------

  login: async (username: string, password: string): Promise<LoginResponse> => {
    const response = await fetch('/api/v1/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });
    return handleResponse<LoginResponse>(response);
  },

  setAuthToken: (token: string | null) => {
    authToken = token;
  },
};

export default api;
