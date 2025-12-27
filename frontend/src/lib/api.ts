/**
 * API Client for EmailOps Cortex Backend
 *
 * Provides typed methods for all backend endpoints with Zod validation.
 */
import { z } from 'zod';
import { logger } from './logger';
import * as s from './schemas';

// Export all inferred types from schemas for component use
export * from './schemas';

// =============================================================================
// Type Definitions (Request Bodies)
// =============================================================================

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface PushDocument {
  document_id?: string;
  source_type?: string;
  text: string;
  metadata?: Record<string, unknown>;
}


// =============================================================================
// API Client
// =============================================================================

let authToken: string | null = null;

const handleResponse = async <T extends z.ZodTypeAny>(
  response: Response,
  schema: T
): Promise<z.infer<T>> => {
  if (response.status === 401) {
    window.dispatchEvent(new Event('unauthorized'));
  }
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
  const data = await response.json();
  try {
    return schema.parse(data);
  } catch (error) {
    logger.error('API response validation failed:', {
        error,
        url: response.url,
        status: response.status,
        data,
    });
    throw new Error(`API response validation failed for ${response.url}.`);
  }
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

  fetchHealth: async () => {
    const response = await fetch('/health');
    return handleResponse(response, s.HealthResponseSchema);
  },

  fetchVersion: async () => {
    const response = await fetch('/version');
    return handleResponse(response, s.VersionSchema);
  },

  // ---------------------------------------------------------------------------
  // RAG Endpoints
  // ---------------------------------------------------------------------------

  search: async (query: string, k = 10, filters: Record<string, unknown> = {}) => {
    const response = await fetch('/api/v1/search', {
      method: 'POST',
      headers: getHeaders(),
      body: JSON.stringify({ query, k, filters }),
    });
    return handleResponse(response, s.SearchResponseSchema);
  },

  ask: async (query: string, threadId?: string, k = 10) => {
    const response = await fetch('/api/v1/answer', {
      method: 'POST',
      headers: getHeaders(),
      body: JSON.stringify({ query, thread_id: threadId, k }),
    });
    return handleResponse(response, s.AnswerResponseSchema);
  },

  chat: async (messages: ChatMessage[], threadId?: string, k = 10) => {
    const response = await fetch('/api/v1/chat', {
      method: 'POST',
      headers: getHeaders(),
      body: JSON.stringify({ messages, thread_id: threadId, k }),
    });
    return handleResponse(response, s.ChatResponseSchema);
  },

  // ---------------------------------------------------------------------------
  // Ingestion Endpoints
  // ---------------------------------------------------------------------------

  listS3Folders: async (prefix = 'Outlook/', limit = 100) => {
    const params = new URLSearchParams({ prefix, limit: String(limit) });
    const response = await fetch(`/api/v1/ingest/list?${params}`);
    return handleResponse(response, s.ListS3FoldersResponseSchema);
  },

  startIngestion: async (prefix = 'Outlook/', limit?: number, dryRun = false) => {
    const response = await fetch('/api/v1/ingest/s3', {
      method: 'POST',
      headers: getHeaders(),
      body: JSON.stringify({ prefix, limit, dry_run: dryRun }),
    });
    return handleResponse(response, s.IngestJobResponseSchema);
  },

  getIngestionStatus: async (jobId: string) => {
    const response = await fetch(`/api/v1/ingest/status/${jobId}`);
    return handleResponse(response, s.IngestStatusResponseSchema);
  },

  pushDocuments: async (documents: PushDocument[], generateEmbeddings = true) => {
    const response = await fetch('/api/v1/ingest/push', {
      method: 'POST',
      headers: getHeaders(),
      body: JSON.stringify({ documents, generate_embeddings: generateEmbeddings }),
    });
    return handleResponse(response, s.PushIngestResponseSchema);
  },

  // ---------------------------------------------------------------------------
  // Admin Endpoints
  // ---------------------------------------------------------------------------

  runDoctor: async () => {
    const response = await fetch('/api/v1/admin/doctor', { method: 'POST' });
    return handleResponse(response, s.DoctorReportSchema);
  },

  fetchStatus: async () => {
    const response = await fetch('/api/v1/admin/status');
    return handleResponse(response, s.SystemStatusSchema);
  },

  fetchConfig: async () => {
    const response = await fetch('/api/v1/admin/config');
    return handleResponse(response, s.SystemConfigSchema);
  },

  // ---------------------------------------------------------------------------
  // Draft & Summarize Endpoints
  // ---------------------------------------------------------------------------

  draftEmail: async (instruction: string, threadId?: string, tone = 'professional') => {
    const response = await fetch('/api/v1/draft', {
      method: 'POST',
      headers: getHeaders(),
      body: JSON.stringify({ instruction, thread_id: threadId, tone }),
    });
    return handleResponse(response, s.DraftEmailResponseSchema);
  },

  summarizeThread: async (threadId: string, maxLength = 500) => {
    const response = await fetch('/api/v1/summarize', {
      method: 'POST',
      headers: getHeaders(),
      body: JSON.stringify({ thread_id: threadId, max_length: maxLength }),
    });
    return handleResponse(response, s.SummarizeResponseSchema);
  },

  // ---------------------------------------------------------------------------
  // Auth Endpoints
  // ---------------------------------------------------------------------------

  login: async (username: string, password: string) => {
    const response = await fetch('/api/v1/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });
    return handleResponse(response, s.LoginResponseSchema);
  },

  setAuthToken: (token: string | null) => {
    authToken = token;
  },
};

export default api;
