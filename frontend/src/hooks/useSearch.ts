import { useQuery } from '@tanstack/react-query';
import { api, type SearchResponse, type ApiError } from '../lib/api';
import { logger } from '../lib/logger';

interface UseSearchParams {
  query: string;
  k?: number;
  filters?: Record<string, unknown>;
}

export const useSearch = ({ query, k = 10, filters = {} }: UseSearchParams) => {
  const {
    data,
    isLoading,
    error,
    isError,
  } = useQuery<SearchResponse, ApiError>({
    queryKey: ['search', query, k, filters],
    queryFn: () => {
      logger.debug('useSearch: Executing search query', { query_length: query.length, k });
      return api.search(query, k, filters);
    },
    enabled: query.length > 0,
  });

  if (isError && error) {
    logger.error('useSearch: Query error', {
      error: error.message,
      status: error.status,
    });
  }

  return { data, isLoading, error, isError };
};
