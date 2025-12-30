import { useMutation } from '@tanstack/react-query';
import { api, type LoginResponse } from '../lib/api';

export const useLogin = () => {
  const mutation = useMutation<
    LoginResponse,
    Error,
    Parameters<typeof api.login>
  >({
    mutationFn: ([username, password]) => api.login(username, password),
    onSuccess: (data) => {
      api.setAuthToken(data.access_token);
    },
  });

  return {
    login: mutation.mutate,
    loginAsync: mutation.mutateAsync,
    data: mutation.data,
    isLoading: mutation.isPending,
    error: mutation.error,
    isSuccess: mutation.isSuccess,
  };
};
