import { useQuery } from '@tanstack/react-query';
import { api, SearchResponse, ApiError } from '../lib/api';

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
    queryFn: () => api.search(query, k, filters),
    enabled: query.length > 0,
  });

  return { data, isLoading, error, isError };
};
