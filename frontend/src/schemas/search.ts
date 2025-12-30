import { z } from 'zod';

/**
 * Schema for a single search result item.
 * This validates the structure of individual search result objects.
 */
export const SearchResultSchema = z.object({
  id: z.string().min(1, 'ID cannot be empty'),
  title: z.string().min(1, 'Title cannot be empty'),
  snippet: z.string(),
  url: z.string().url('Invalid URL format'),
});

// Infer the TypeScript type from the SearchResultSchema
export type SearchResult = z.infer<typeof SearchResultSchema>;

/**
 * Schema for the search filters.
 * This validates the structure of the search filters object.
 */
export const SearchFiltersSchema = z.object({
  filetype: z.string().optional(),
  author: z.string().optional(),
  dateRange: z
    .object({
      startDate: z.string().optional(),
      endDate: z.string().optional(),
    })
    .optional(),
});

// Infer the TypeScript type from the SearchFiltersSchema
export type SearchFilters = z.infer<typeof SearchFiltersSchema>;

/**
 * Schema for the search request.
 * This validates the structure of the search request object sent to the API.
 */
export const SearchRequestSchema = z.object({
  query: z.string().min(1, 'Query cannot be empty'),
  filters: SearchFiltersSchema.optional(),
});

// Infer the TypeScript type from the SearchRequestSchema
export type SearchRequest = z.infer<typeof SearchRequestSchema>;

/**
 * Schema for the search response from the API.
 * This validates the structure of the data received from the /search endpoint.
 */
export const SearchResponseSchema = z.object({
  results: z.array(SearchResultSchema),
  total: z.number().int().min(0, 'Total must be a non-negative integer'),
  page: z.number().int().min(1, 'Page must be a positive integer'),
  pageSize: z.number().int().min(1, 'Page size must be a positive integer'),
});

// Infer the TypeScript type from the SearchResponseSchema
export type SearchResponse = z.infer<typeof SearchResponseSchema>;
