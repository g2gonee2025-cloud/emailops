import { useMutation } from '@tanstack/react-query';
import { api, type DraftEmailResponse, type ApiError } from '../lib/api';

interface DraftEmailVariables {
  instruction: string;
  threadId?: string;
  tone?: string;
}

/**
 * Custom hook to draft an email using the API.
 *
 * This hook encapsulates the logic for calling the `api.draftEmail` method
 * and manages the mutation state (loading, error, data) via TanStack Query.
 *
 * @returns {object} The result of the `useMutation` hook, which includes:
 *  - `mutate`: The function to trigger the email draft mutation.
 *  - `data`: The response from the API upon successful mutation.
 *  - `error`: The error object if the mutation fails.
 *  - `isPending`: A boolean indicating if the mutation is currently in progress.
 *  - ...other properties from `useMutation`.
 */
export const useDraftEmail = () => {
  return useMutation<DraftEmailResponse, ApiError, DraftEmailVariables>({
    mutationFn: ({ instruction, threadId, tone }: DraftEmailVariables) =>
      api.draftEmail(instruction, threadId, tone),
  });
};
