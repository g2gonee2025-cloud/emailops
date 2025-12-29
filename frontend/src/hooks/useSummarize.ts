import { useMutation } from '@tanstack/react-query';
import { api } from '../lib/api';

export const useSummarize = () => {
  const mutation = useMutation({
    mutationFn: ({ threadId, maxLength }: { threadId: string; maxLength?: number }) =>
      api.summarizeThread(threadId, maxLength),
  });

  return {
    summarize: mutation.mutate,
    data: mutation.data,
    isLoading: mutation.isPending,
    error: mutation.error,
    ...mutation,
  };
};
