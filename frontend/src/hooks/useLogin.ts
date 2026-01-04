import { useMutation } from '@tanstack/react-query';
import { api, type LoginResponse } from '../lib/api';
import { useAuth } from '../contexts/AuthContext';

export const useLogin = () => {
  const { setTokensFromResponse } = useAuth();

  const mutation = useMutation<
    LoginResponse,
    Error,
    Parameters<typeof api.login>
  >({
    mutationFn: ([username, password]) => api.login(username, password),
    onSuccess: (data) => {
      setTokensFromResponse(data);
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
