/** @vitest-environment jsdom */
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { describe, it, expect, vi } from 'vitest';
import { useLogin } from './useLogin';
import { api } from '@/lib/api';

// Mock the api module
vi.mock('@/lib/api', () => ({
  api: {
    login: vi.fn(),
  },
}));

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false, // Disable retries for tests
      },
    },
  });
  // eslint-disable-next-line react/display-name
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe('useLogin', () => {
  it('should return initial state correctly', () => {
    const { result } = renderHook(() => useLogin(), { wrapper: createWrapper() });

    expect(result.current.data).toBeUndefined();
    expect(result.current.isLoading).toBe(false);
    expect(result.current.error).toBe(null);
  });

  it('should handle successful login', async () => {
    const loginData = { access_token: 'test-token', token_type: 'bearer', expires_in: 3600 };
    (api.login as vi.Mock).mockResolvedValue(loginData);

    const { result } = renderHook(() => useLogin(), { wrapper: createWrapper() });

    result.current.login({ username: 'testuser', password: 'password' });

    await waitFor(() => expect(result.current.data).toBeDefined());

    expect(result.current.data).toEqual(loginData);
    expect(result.current.isLoading).toBe(false);
    expect(result.current.error).toBe(null);
  });

  it('should handle failed login', async () => {
    const error = new Error('Invalid credentials');
    (api.login as vi.Mock).mockRejectedValue(error);

    const { result } = renderHook(() => useLogin(), { wrapper: createWrapper() });

    result.current.login({ username: 'testuser', password: 'wrongpassword' });

    await waitFor(() => expect(result.current.error).toBeDefined());

    expect(result.current.data).toBeUndefined();
    expect(result.current.isLoading).toBe(false);
    expect(result.current.error).toEqual(error);
  });
});
