import { z } from 'zod';

/**
 * Schema for a single search result object.
 */
export const searchResultSchema = z.object({
  chunk_id: z.string().uuid('Invalid chunk ID'),
  conversation_id: z.string().uuid('Invalid conversation ID'),
  thread_id: z.string().uuid('Invalid thread ID').optional(),
  content: z.string(),
  score: z.number().min(0, 'Score must be non-negative'),
  metadata: z.record(z.unknown()).optional(),
});

/**
 * TypeScript type inferred from the search result schema.
 */
export type SearchResult = z.infer<typeof searchResultSchema>;


/**
 * Schema for the full search API response.
 */
export const searchResponseSchema = z.object({
  correlation_id: z.string().optional(),
  results: z.array(searchResultSchema),
  total_count: z.number().int('Total count must be an integer').min(0),
  query_time_ms: z.number().min(0, 'Query time must be non-negative'),
});

/**
 * TypeScript type inferred from the search response schema.
 */
export type SearchResponse = z.infer<typeof searchResponseSchema>;


// =======================================================================
// Helper Validators
// =======================================================================

/**
 * Validates if the given data is a valid SearchResult object.
 * @param data The data to validate.
 * @returns True if the data is a valid SearchResult, false otherwise.
 */
export const isSearchResult = (data: unknown): data is SearchResult => {
  return searchResultSchema.safeParse(data).success;
};

/**
 * Validates if the given data is a valid SearchResponse object.
 * @param data The data to validate.
 * @returns True if the data is a valid SearchResponse, false otherwise.
 */
export const isSearchResponse = (data: unknown): data is SearchResponse => {
  return searchResponseSchema.safeParse(data).success;
};
