import { useQuery } from '@tanstack/react-query';
import { api } from '../lib/api';

export const useDashboardMetrics = () => {
  const {
    data: metrics,
    isLoading,
    error,
  } = useQuery({
    queryKey: ['dashboardMetrics'],
    queryFn: api.getDashboardMetrics,
  });

  return {
    metrics,
    isLoading,
    error,
  };
};
