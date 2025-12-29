import { z } from 'zod';

// =============================================================================
// Search Result Schema
// =============================================================================

export const SearchResultSchema = z.object({
  chunk_id: z.string().uuid(),
  conversation_id: z.string(),
  thread_id: z.string().optional(),
  content: z.string(),
  score: z.number(),
  metadata: z.record(z.unknown()).optional(),
});

export type SearchResult = z.infer<typeof SearchResultSchema>;

// =============================================================================
// Thread Summary Schema
// =============================================================================

export const ThreadSummarySchema = z.object({
  summary: z.string(),
  key_points: z.array(z.string()).optional(),
  action_items: z.array(z.string()).optional(),
});

export type ThreadSummary = z.infer<typeof ThreadSummarySchema>;
