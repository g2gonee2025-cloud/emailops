import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';

/**
 * A hook for interacting with a specific thread, providing both data fetching and mutation capabilities.
 *
 * @param threadId The ID of the thread to interact with. If null or undefined, the query will be disabled.
 * @returns An object containing the thread data, loading/error states, and mutation helpers.
 */
export const useThread = (threadId: string | null | undefined) => {
  const queryClient = useQueryClient();

  const {
    data,
    isLoading,
    error: queryError,
  } = useQuery({
    queryKey: ['thread', threadId],
    queryFn: () => {
      if (!threadId) {
        return Promise.reject(new Error('threadId is required'));
      }
      return api.fetchThread(threadId);
    },
    enabled: !!threadId, // The query will not run until a threadId is provided
  });

  /**
   * Mutation for summarizing the current thread.
   */
  const useSummarizeThread = () => {
    return useMutation({
      mutationFn: (maxLength?: number) => {
        if (!threadId) {
          return Promise.reject(new Error('threadId is required for summarization'));
        }
        return api.summarizeThread(threadId, maxLength);
      },
      onSuccess: () => {
        // After summarizing, invalidate related queries to refetch fresh data if needed
        queryClient.invalidateQueries({ queryKey: ['thread', threadId, 'summary'] });
      },
    });
  };

  const {
    mutate: summarize,
    isPending: isSummarizing,
    error: summarizeError,
    data: summaryData,
  } = useSummarizeThread();

  return {
    // Query state
    data,
    isLoading,
    error: queryError,

    // Summarize mutation
    summarize,
    summaryData,
    isSummarizing,
    summarizeError,
  };
};