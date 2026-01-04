import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api, type IngestStatusResponse } from '@/lib/api';

const INGESTION_QUERY_KEY = 'ingestion';

const TERMINAL_STATUSES = ['completed', 'failed', 'completed_with_errors'] as const;
const POLLING_INTERVAL_MS = 2000;

/**
 * Hook for interacting with the ingestion API.
 */
export const useIngestion = () => {
  const queryClient = useQueryClient();

  const {
    data: folders,
    isLoading: isLoadingFolders,
    error: foldersError,
  } = useQuery({
    queryKey: [INGESTION_QUERY_KEY, 'folders'],
    queryFn: () => api.listS3Folders(),
  });

  const {
    mutate: startIngestion,
    isPending: isStartingIngestion,
    error: startIngestionError,
  } = useMutation({
    mutationFn: (variables: { prefix?: string; limit?: number; dryRun?: boolean }) =>
      api.startIngestion(variables.prefix, variables.limit, variables.dryRun),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [INGESTION_QUERY_KEY] });
    },
  });

  const useIngestionStatus = (
    jobId: string,
    enabled: boolean,
    refetchInterval = 5000,
  ) => {
    return useQuery({
      queryKey: [INGESTION_QUERY_KEY, 'status', jobId],
      queryFn: () => api.getIngestionStatus(jobId),
      enabled,
      refetchInterval,
    });
  };

  return {
    folders,
    isLoadingFolders,
    foldersError,
    startIngestion,
    isStartingIngestion,
    startIngestionError,
    useIngestionStatus,
  };
};

/**
 * Standalone hook for polling ingestion job status.
 * Polls every 2s while status is "started" or "processing".
 * Stops polling when status is "completed", "failed", or "completed_with_errors".
 *
 * @param jobId - The job ID to poll status for, or null to disable polling
 * @returns { data, isLoading, error, refetch }
 */
export const useIngestionStatus = (jobId: string | null) => {
  const { data, isLoading, error, refetch } = useQuery<IngestStatusResponse>({
    queryKey: [INGESTION_QUERY_KEY, 'status', jobId],
    queryFn: ({ signal }) => api.getIngestionStatus(jobId!, { signal }),
    enabled: jobId !== null,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (!status) return POLLING_INTERVAL_MS;
      if (TERMINAL_STATUSES.includes(status as (typeof TERMINAL_STATUSES)[number])) {
        return false;
      }
      return POLLING_INTERVAL_MS;
    },
  });

  return { data, isLoading, error, refetch };
};
