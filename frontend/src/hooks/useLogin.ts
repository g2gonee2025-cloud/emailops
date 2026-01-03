import { useMutation } from '@tanstack/react-query';
import { api, type LoginResponse } from '../lib/api';
import { logger } from '../lib/logger';

export const useLogin = () => {
  const mutation = useMutation<
    LoginResponse,
    Error,
    Parameters<typeof api.login>
  >({
    mutationFn: ([username, password]) => {
      logger.info('useLogin: Login mutation started', { username });
      return api.login(username, password);
    },
    onSuccess: (data, [username]) => {
      logger.info('useLogin: Login successful', {
        username,
        token_type: data.token_type,
        expires_in: data.expires_in,
      });
      api.setAuthToken(data.access_token);
    },
    onError: (error, [username]) => {
      logger.error('useLogin: Login failed', {
        username,
        error: error.message,
      });
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
