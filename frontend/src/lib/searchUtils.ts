/**
 * Utilities for serializing and parsing search query parameters from a URL.
 * This allows for the search state to be stored in and read from the URL,
 * enabling shareable and bookmarkable search result pages.
 */

import { z } from 'zod';
import { logger } from './logger';

// ========================================================================================
// Zod Schemas for Validation
// ========================================================================================

/**
 * Defines the structure and validation rules for search parameters.
 */
const SearchParamsSchema = z.object({
  query: z.string().default(''),
  k: z.coerce
    .number()
    .transform((val) => Math.trunc(val))
    .pipe(z.number().positive())
    .default(10),
  filters: z
    .record(z.unknown())
    .default({})
    .refine((val) => typeof val === 'object' && !Array.isArray(val), {
      message: 'Filters must be an object.',
    }),
});

export type SearchParams = z.infer<typeof SearchParamsSchema>;

// ========================================================================================
// Public Functions
// ========================================================================================

/**
 * Serializes a search parameters object into a URL query string.
 *
 * @param params - The search parameters object to serialize.
 * @returns A URL query string (e.g., "?query=test&k=20&filters=%7B%22author%22%3A%22jane%22%7D").
 */
export function serializeSearchQuery(params: Partial<SearchParams>): string {
  const searchParams = new URLSearchParams();

  if (params.query) {
    searchParams.set('query', params.query);
  }

  if (params.k) {
    searchParams.set('k', String(params.k));
  }

  if (params.filters && Object.keys(params.filters).length > 0) {
    try {
      const filtersString = JSON.stringify(params.filters);
      searchParams.set('filters', filtersString);
    } catch (error) {
      logger.error('Failed to serialize search filters', { error, filters: params.filters });
    }
  }

  const queryString = searchParams.toString();
  return queryString ? `?${queryString}` : '';
}

/**
 * Parses a URL query string into a validated search parameters object.
 *
 * @param queryString - The URL query string to parse (e.g., window.location.search).
 * @returns A validated search parameters object with default values for missing fields.
 */
export function parseSearchQuery(queryString: string): SearchParams {
  const searchParams = new URLSearchParams(queryString);
  const filtersParam = searchParams.get('filters');

  let parsedFilters: Record<string, unknown> = {};
  if (filtersParam) {
    try {
      parsedFilters = JSON.parse(filtersParam);
      if (typeof parsedFilters !== 'object' || Array.isArray(parsedFilters) || parsedFilters === null) {
        throw new Error('Parsed filters are not a valid object.');
      }
    } catch (error) {
      logger.warn('Failed to parse search filters from URL, using default.', {
        error,
        filtersParam,
      });
      parsedFilters = {};
    }
  }

  const rawParams: { query?: unknown; k?: unknown; filters?: unknown } = {
    filters: parsedFilters,
  };
  if (searchParams.has('query')) {
    rawParams.query = searchParams.get('query');
  }
  if (searchParams.has('k')) {
    rawParams.k = searchParams.get('k');
  }

  const validationResult = SearchParamsSchema.safeParse(rawParams);

  if (!validationResult.success) {
    logger.warn('URL search parameters failed validation, using defaults for invalid fields.', {
      errors: validationResult.error.flatten(),
      rawParams,
    });

    const fieldErrors = validationResult.error.flatten().fieldErrors;
    const paramsToParse: typeof rawParams = {};

    if (!fieldErrors.query && 'query' in rawParams) {
      paramsToParse.query = rawParams.query;
    }
    if (!fieldErrors.k && 'k' in rawParams) {
      paramsToParse.k = rawParams.k;
    }
    if (!fieldErrors.filters && 'filters' in rawParams) {
      paramsToParse.filters = rawParams.filters;
    }

    return SearchParamsSchema.parse(paramsToParse);
  }

  return validationResult.data;
}
