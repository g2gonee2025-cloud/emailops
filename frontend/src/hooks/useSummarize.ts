import { useMutation } from '@tanstack/react-query';
import { api, type SummarizeResponse, type ApiError } from '../lib/api';

interface SummarizeVariables {
  threadId: string;
  maxLength?: number;
}

export const useSummarize = () => {
  const mutation = useMutation<
    SummarizeResponse,
    ApiError,
    SummarizeVariables
  >({
    mutationFn: ({ threadId, maxLength }: SummarizeVariables) =>
      api.summarizeThread(threadId, maxLength),
  });

  return {
    summarize: mutation.mutate,
    summarizeAsync: mutation.mutateAsync,
    data: mutation.data,
    isLoading: mutation.isPending,
    error: mutation.error,
    status: mutation.status,
  };
};
