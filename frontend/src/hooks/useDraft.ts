import { useMutation, UseMutationResult } from '@tanstack/react-query';
import { api, DraftEmailResponse } from '@/lib/api';

interface DraftVariables {
  instruction: string;
  threadId?: string;
  tone?: string;
}

/**
 * A hook for drafting emails using the API.
 *
 * @returns A mutation object with the following properties:
 *  - `data`: The data returned from the API.
 *  - `isPending`: `true` if the mutation is pending.
 *  - `error`: The error object if the mutation fails.
 *  - `mutate`: A function to trigger the mutation.
 *  - `mutateAsync`: An async function to trigger the mutation.
 */
export const useDraft = (): UseMutationResult<
  DraftEmailResponse,
  Error,
  DraftVariables
> => {
  return useMutation<DraftEmailResponse, Error, DraftVariables>({
    mutationFn: ({ instruction, threadId, tone }) =>
      api.draftEmail(instruction, threadId, tone),
  });
};
