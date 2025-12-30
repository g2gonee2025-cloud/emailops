import { z } from 'zod';

/**
 * @file Zod schemas for Search API endpoints.
 * @description These schemas validate the data structures used in search operations.
 */

/**
 * Schema for a single search result item returned by the API.
 * This ensures each result has the necessary fields and correct data types.
 */
export const SearchResultSchema = z.object({
  chunk_id: z.string().min(1, 'Chunk ID cannot be empty'),
  conversation_id: z.string().min(1, 'Conversation ID cannot be empty'),
  thread_id: z.string().optional(),
  content: z.string(),
  score: z.number().min(0, 'Score must be non-negative'),
  metadata: z.record(z.unknown()).optional(),
});

// Infer the TypeScript type from the SearchResultSchema for use in components.
export type SearchResult = z.infer<typeof SearchResultSchema>;

/**
 * Schema for the entire search response object from the API.
 * This validates the top-level structure of the `/search` endpoint's payload.
 */
export const SearchResponseSchema = z.object({
  correlation_id: z.string().optional(),
  results: z.array(SearchResultSchema),
  total_count: z.number().int().min(0, 'Total count must be a non-negative integer'),
  query_time_ms: z.number().min(0, 'Query time must be a non-negative number'),
});

// Infer the TypeScript type from the SearchResponseSchema for use with TanStack Query.
export type SearchResponse = z.infer<typeof SearchResponseSchema>;
