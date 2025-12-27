import { z } from 'zod';

// =============================================================================
// Base Schemas
// =============================================================================

export const SearchResultSchema = z.object({
  chunk_id: z.string(),
  conversation_id: z.string(),
  thread_id: z.string().optional(),
  content: z.string(),
  score: z.number(),
  metadata: z.record(z.unknown()).optional(),
});

export const AnswerSchema = z.object({
  text: z.string(),
  confidence_overall: z.number(),
  sources: z
    .array(
      z.object({
        chunk_id: z.string(),
        content: z.string(),
        score: z.number(),
      })
    )
    .optional(),
});

export const EmailDraftSchema = z.object({
  subject: z.string(),
  body: z.string(),
  to: z.array(z.string()).optional(),
  cc: z.array(z.string()).optional(),
});

export const ThreadSummarySchema = z.object({
    summary: z.string(),
    key_points: z.array(z.string()).optional(),
    action_items: z.array(z.string()).optional(),
});


// =============================================================================
// API Response Schemas
// =============================================================================

export const VersionSchema = z.record(z.string());

export const SearchResponseSchema = z.object({
  correlation_id: z.string().optional(),
  results: z.array(SearchResultSchema),
  total_count: z.number(),
  query_time_ms: z.number(),
});

export const AnswerResponseSchema = z.object({
  correlation_id: z.string().optional(),
  answer: AnswerSchema,
  confidence: z.number(),
  debug_info: z.record(z.unknown()).optional(),
});

export const ChatResponseSchema = z.object({
    correlation_id: z.string().optional(),
    action: z.enum(['answer', 'search', 'summarize']),
    reply: z.string(),
    answer: AnswerSchema.optional(),
    search_results: z.array(SearchResultSchema).optional(),
    debug_info: z.record(z.unknown()).optional(),
});

export const DraftEmailResponseSchema = z.object({
    correlation_id: z.string().optional(),
    draft: EmailDraftSchema,
    confidence: z.number(),
    iterations: z.number(),
});

export const SummarizeResponseSchema = z.object({
    correlation_id: z.string().optional(),
    summary: ThreadSummarySchema,
});

export const PushIngestResponseSchema = z.object({
    job_id: z.string(),
    documents_received: z.number(),
    documents_ingested: z.number(),
    chunks_created: z.number(),
    embeddings_generated: z.number(),
    errors: z.array(z.string()),
    message: z.string(),
});

export const LoginResponseSchema = z.object({
    access_token: z.string(),
    token_type: z.string(),
    expires_in: z.number(),
});

export const S3FolderSchema = z.object({
  folder: z.string(),
  size_bytes: z.number().optional(),
  last_modified: z.string().optional(),
});

export const ListS3FoldersResponseSchema = z.object({
  prefix: z.string(),
  folders: z.array(S3FolderSchema),
  count: z.number(),
});

export const IngestJobResponseSchema = z.object({
  job_id: z.string(),
  status: z.string(),
  folders_found: z.number().optional(),
  folders_to_process: z.array(z.string()).optional(),
  message: z.string(),
});

export const IngestStatusResponseSchema = z.object({
  job_id: z.string(),
  status: z.string(),
  folders_processed: z.number(),
  threads_created: z.number(),
  chunks_created: z.number(),
  embeddings_generated: z.number(),
  errors: z.number(),
  skipped: z.number(),
  message: z.string(),
});

export const DoctorCheckSchema = z.object({
  name: z.string(),
  status: z.enum(['pass', 'fail', 'warn']),
  message: z.string().optional(),
  details: z.record(z.unknown()).optional(),
});

export const DoctorReportSchema = z.object({
  overall_status: z.enum(['healthy', 'degraded', 'unhealthy']),
  checks: z.array(DoctorCheckSchema),
});

export const SystemStatusSchema = z.object({
  status: z.string(),
  service: z.string(),
  env: z.string(),
});

export const SystemConfigSchema = z.object({
  environment: z.string(),
  provider: z.string(),
  log_level: z.string(),
  database_url: z.string().optional(),
});

export const HealthResponseSchema = z.object({
  status: z.string(),
  version: z.string(),
  environment: z.string(),
});

// =============================================================================
// Inferred Types
// =============================================================================

export type SearchResult = z.infer<typeof SearchResultSchema>;
export type SearchResponse = z.infer<typeof SearchResponseSchema>;
export type Answer = z.infer<typeof AnswerSchema>;
export type AnswerResponse = z.infer<typeof AnswerResponseSchema>;
export type ChatResponse = z.infer<typeof ChatResponseSchema>;
export type EmailDraft = z.infer<typeof EmailDraftSchema>;
export type DraftEmailResponse = z.infer<typeof DraftEmailResponseSchema>;
export type ThreadSummary = z.infer<typeof ThreadSummarySchema>;
export type SummarizeResponse = z.infer<typeof SummarizeResponseSchema>;
export type PushIngestResponse = z.infer<typeof PushIngestResponseSchema>;
export type LoginResponse = z.infer<typeof LoginResponseSchema>;
export type S3Folder = z.infer<typeof S3FolderSchema>;
export type ListS3FoldersResponse = z.infer<typeof ListS3FoldersResponseSchema>;
export type IngestJobResponse = z.infer<typeof IngestJobResponseSchema>;
export type IngestStatusResponse = z.infer<typeof IngestStatusResponseSchema>;
export type DoctorCheck = z.infer<typeof DoctorCheckSchema>;
export type DoctorReport = z.infer<typeof DoctorReportSchema>;
export type SystemStatus = z.infer<typeof SystemStatusSchema>;
export type SystemConfig = z.infer<typeof SystemConfigSchema>;
export type HealthResponse = z.infer<typeof HealthResponseSchema>;
