import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';

const INGESTION_QUERY_KEY = 'ingestion';

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
      refetchInterval, // Poll every 5 seconds
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
