/**
 * API Client for EmailOps Cortex Backend
 *
 * Provides typed methods for all backend endpoints.
 */

import { z } from 'zod';
import { GeneratedDraftSchema } from '../schemas/draft';
import { logger } from './logger';
import {
  doctorReportSchema,
  statusDataSchema,
  type DoctorReport,
  type StatusData,
} from '../schemas/admin';

export type { DoctorReport, StatusData };

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

const getHeaders = (): HeadersInit => {
  const headers: HeadersInit = { 'Content-Type': 'application/json' };
  const authToken = localStorage.getItem('auth_token');
  if (authToken) {
    headers['Authorization'] = `Bearer ${authToken}`;
  }
  return headers;
};

export const request = async <T>(
  endpoint: string,
  options: RequestInit = {},
  includeAuth = true,
): Promise<T> => {
  const baseHeaders = includeAuth ? getHeaders() : {};

  const config: RequestInit = {
    ...options,
    headers: {
      ...baseHeaders,
      ...options.headers,
    },
  };

  try {
    const url = endpoint.startsWith('http') ? endpoint : `${API_BASE_URL}${endpoint}`;
    const response = await fetch(url, config);

    if (!response.ok) {
      let errorDetails;
      try {
        errorDetails = await response.json();
      } catch (_e) {
        errorDetails = { detail: response.statusText };
      }

      const apiError = new ApiError(
        `API request failed: ${response.status}`,
        response.status,
        errorDetails,
      );

      if (response.status === 401 && !response.url.includes('login')) {
        window.dispatchEvent(new CustomEvent('cortex-unauthorized'));
      } else if (response.status !== 401) {
        window.dispatchEvent(new CustomEvent('api:error', { detail: apiError }));
      }

      throw apiError;
    }

    if (response.status === 204) {
      return {} as T;
    }

    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    logger.error('Network or other fetch error occurred.', {
      endpoint: endpoint,
      error: error instanceof Error ? error.message : String(error),
    });
    throw new Error('A network error occurred.');
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
