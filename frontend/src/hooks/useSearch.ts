import { useQuery } from '@tanstack/react-query';
import { api, SearchResponse } from '@/lib/api';

interface UseSearchParams {
  query: string;
  k?: number;
  filters?: Record<string, unknown>;
  enabled?: boolean;
}

export const useSearch = ({ query, k = 10, filters = {}, enabled = true }: UseSearchParams) => {
  const {
    data,
    isLoading,
    error,
  } = useQuery<SearchResponse, Error>({
    queryKey: ['search', { query, k, filters }],
    queryFn: () => api.search(query, k, filters),
    enabled: enabled && !!query,
  });

  return {
    data,
    isLoading,
    error,
  };
};
