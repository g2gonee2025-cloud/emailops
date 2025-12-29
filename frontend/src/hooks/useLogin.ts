import { useMutation } from '@tanstack/react-query';
import { api } from '@/lib/api';

type LoginCredentials = {
  username: Parameters<typeof api.login>[0];
  password: Parameters<typeof api.login>[1];
};

/**
 * @description
 * A hook to handle the login mutation.
 *
 * @returns {object}
 * - `login`: A function to trigger the login mutation.
 * - `data`: The data returned from the API on success.
 * - `isLoading`: A boolean indicating if the mutation is in progress.
 * - `error`: An error object if the mutation fails.
 */
export const useLogin = () => {
  const {
    mutate: login,
    data,
    isPending: isLoading,
    error,
  } = useMutation({
    mutationFn: ({ username, password }: LoginCredentials) =>
      api.login(username, password),
  });

  return { login, data, isLoading, error };
};
