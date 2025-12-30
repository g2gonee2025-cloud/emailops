import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../lib/api';

export const useDashboardMetrics = (jobId?: string) => {
  const queryClient = useQueryClient();

  const statusQuery = useQuery({
    queryKey: ['systemStatus'],
    queryFn: () => api.fetchStatus(),
  });

  const configQuery = useQuery({
    queryKey: ['systemConfig'],
    queryFn: () => api.fetchConfig(),
  });

  const doctorQuery = useQuery({
    queryKey: ['doctorReport'],
    queryFn: () => api.runDoctor(),
    enabled: false, // Typically run on-demand
  });

  const ingestionStatusQuery = useQuery({
    queryKey: ['ingestionStatus', jobId],
    queryFn: () => api.getIngestionStatus(jobId as string),
    enabled: !!jobId, // Only run if a jobId is provided
    refetchInterval: (query) => {
      const data = query.state.data;
      if (data?.status === 'completed' || data?.status === 'failed') {
        return false;
      }
      return 5000; // Poll every 5 seconds while in progress
    },
  });

  const startIngestionMutation = useMutation({
    mutationFn: (variables: { prefix: string; limit?: number; dryRun?: boolean }) =>
      api.startIngestion(variables.prefix, variables.limit, variables.dryRun),
    onSuccess: () => {
      // Invalidate and refetch relevant queries after a successful mutation
      queryClient.invalidateQueries({ queryKey: ['ingestionStatus'] });
    },
  });

  return {
    systemStatus: {
      data: statusQuery.data,
      isLoading: statusQuery.isLoading,
      error: statusQuery.error,
    },
    systemConfig: {
      data: configQuery.data,
      isLoading: configQuery.isLoading,
      error: configQuery.error,
    },
    doctorReport: {
      data: doctorQuery.data,
      isLoading: doctorQuery.isLoading,
      error: doctorQuery.error,
      runCheck: doctorQuery.refetch,
    },
    ingestionStatus: {
      data: ingestionStatusQuery.data,
      isLoading: ingestionStatusQuery.isLoading,
      error: ingestionStatusQuery.error,
    },
    startIngestion: {
      mutate: startIngestionMutation.mutate,
      mutateAsync: startIngestionMutation.mutateAsync,
      isPending: startIngestionMutation.isPending,
      error: startIngestionMutation.error,
      data: startIngestionMutation.data,
    },
  };
};
