import { renderHook, act, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useLogin } from '../../hooks/useLogin';
import { describe, it, expect, vi, beforeEach, type Mock } from 'vitest';
import { api, ApiError } from '../../lib/api';

// Mock setTokensFromResponse function
const mockSetTokensFromResponse = vi.fn();

// Mock the api module
vi.mock('../../lib/api', () => ({
  api: {
    login: vi.fn(),
    setAuthToken: vi.fn(),
  },
  ApiError: class extends Error {
    status: number;
    constructor(message: string, status: number) {
      super(message);
      this.status = status;
    }
  },
}));

// Mock the AuthContext
vi.mock('../../contexts/AuthContext', () => ({
  useAuth: () => ({
    setTokensFromResponse: mockSetTokensFromResponse,
  }),
}));

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe('useLogin', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should call api.login and set tokens on success', async () => {
    const { result } = renderHook(() => useLogin(), { wrapper: createWrapper() });
    const mockToken = { access_token: 'test-token', refresh_token: 'test-refresh', token_type: 'bearer', expires_in: 3600 };
    (api.login as Mock).mockResolvedValue(mockToken);

    act(() => {
      result.current.loginAsync(['testuser', 'password']);
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(api.login).toHaveBeenCalledWith('testuser', 'password');
    expect(result.current.data).toEqual(mockToken);
    expect(mockSetTokensFromResponse).toHaveBeenCalledWith(mockToken);
  });

  it('should handle login failure', async () => {
    const { result } = renderHook(() => useLogin(), { wrapper: createWrapper() });
    const mockError = new ApiError('Invalid credentials', 401);
    (api.login as Mock).mockRejectedValue(mockError);

    act(() => {
      result.current.loginAsync(['testuser', 'wrongpassword']).catch(() => {
        /* ignore */
      });
    });

    await waitFor(() => {
      expect(result.current.error).not.toBeNull();
    });

    expect(api.login).toHaveBeenCalledWith('testuser', 'wrongpassword');
    expect(result.current.isSuccess).toBe(false);
    expect(result.current.error).toEqual(mockError);
    expect(mockSetTokensFromResponse).not.toHaveBeenCalled();
  });
});
