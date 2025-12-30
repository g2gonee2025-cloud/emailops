import { type SearchParams, SearchParamsSchema } from '../schemas/search';
import { logger } from './logger';

/**
 * @file Utilities for serializing and parsing search state to/from URL query parameters.
 * This allows for shareable and bookmarkable search result pages.
 */

/**
 * Serializes a search parameters object into a URLSearchParams string.
 * This function takes a structured SearchParams object and converts it into a string
 * suitable for appending to a URL, enabling the search state to be shared.
 *
 * - The `query` is directly set.
 * - The `k` value (number of results) is converted to a string.
 * - The `filters` object is stringified as JSON.
 *
 * @param {SearchParams} params - The search parameters object to serialize.
 * @returns {URLSearchParams} A URLSearchParams object ready to be used in a URL.
 */
export function serializeSearchToURL(params: SearchParams): URLSearchParams {
  const searchParams = new URLSearchParams();
  if (params.query) {
    searchParams.set('query', params.query);
  }
  if (params.k) {
    searchParams.set('k', String(params.k));
  }
  if (params.filters && Object.keys(params.filters).length > 0) {
    try {
      searchParams.set('filters', JSON.stringify(params.filters));
    } catch (error) {
      logger.error('Failed to stringify search filters', { error });
    }
  }
  return searchParams;
}

/**
 * Parses search parameters from a URLSearchParams object and validates them.
 * This function takes URL query parameters, attempts to parse them back into a
 * structured SearchParams object, and validates the result against the schema.
 * It provides robust handling for missing, malformed, or invalid data, ensuring
 * that the application's search state is always well-formed.
 *
 * - It safely handles numeric conversion for `k`.
 * - It safely parses the JSON string for `filters`.
 * - If parsing fails, it logs a warning and returns the default search parameters.
 *
 * @param {URLSearchParams} searchParams - The URL search parameters to parse.
 * @returns {SearchParams} A validated search parameters object.
 */
export function parseSearchFromURL(searchParams: URLSearchParams): SearchParams {
  const query = searchParams.get('query') || undefined;
  const k = searchParams.get('k');
  const filters = searchParams.get('filters');

  // Build a raw object with values from the URL, but don't parse them yet.
  // Zod will handle coercion and validation.
  const rawParams: {
    query?: string;
    k?: number;
    filters?: Record<string, unknown>;
  } = {};

  if (query) {
    rawParams.query = query;
  }

  if (k) {
    const parsedK = parseInt(k, 10);
    if (!isNaN(parsedK)) {
      rawParams.k = parsedK;
    }
  }

  if (filters) {
    try {
      const parsedFilters = JSON.parse(filters);
      // Ensure it's a non-null object before assigning
      if (typeof parsedFilters === 'object' && parsedFilters !== null) {
        rawParams.filters = parsedFilters;
      } else {
        logger.warn('Parsed filters from URL is not a valid object', {
          parsedFilters,
        });
      }
    } catch (error) {
      logger.warn('Failed to parse search filters JSON from URL', {
        filtersValue: filters,
        error,
      });
    }
  }

  // Use the schema to parse and validate, which also applies defaults
  const parseResult = SearchParamsSchema.safeParse(rawParams);

  if (!parseResult.success) {
    logger.warn('Invalid search params in URL after parsing', {
      errors: parseResult.error.flatten(),
      rawParams,
    });
    // On failure, return a default state by parsing an empty object
    return SearchParamsSchema.parse({});
  }

  return parseResult.data;
}
