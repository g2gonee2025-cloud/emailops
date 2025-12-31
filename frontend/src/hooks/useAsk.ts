import { useMutation } from '@tanstack/react-query';
import { api } from '@/lib/api';
import type { AnswerResponse } from '@/lib/api';

interface AskVariables {
  query: string;
  threadId?: string;
  k?: number;
}

export const useAsk = () => {
  const mutation = useMutation<AnswerResponse, Error, AskVariables>({
    mutationFn: ({ query, threadId, k }) => api.ask(query, threadId, k),
  });

  return {
    ask: mutation.mutate,
    ...mutation,
  };
};
