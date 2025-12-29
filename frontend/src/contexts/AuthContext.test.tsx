
import { renderHook, act } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { vi } from 'vitest';
import { AuthProvider, useAuth } from './AuthContext';
import { api, ApiError } from '@/lib/api';

// Mock the api module
vi.mock('@/lib/api', () => ({
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

// Mock useNavigate
const mockedNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mockedNavigate,
  };
});

describe('AuthContext', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
  });

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <BrowserRouter>
      <AuthProvider>{children}</AuthProvider>
    </BrowserRouter>
  );

  it('should start with an unauthenticated state', () => {
    const { result } = renderHook(() => useAuth(), { wrapper });
    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.token).toBeNull();
  });

  it('should successfully log in and update state', async () => {
    const { result } = renderHook(() => useAuth(), { wrapper });

    const mockToken = 'test_token';
    (api.login as vi.Mock).mockResolvedValue({ access_token: mockToken });

    await act(async () => {
      await result.current.login('testuser', 'password');
    });

    expect(result.current.isAuthenticated).toBe(true);
    expect(result.current.token).toBe(mockToken);
    expect(localStorage.getItem('auth_token')).toBe(mockToken);
    expect(api.setAuthToken).toHaveBeenCalledWith(mockToken);
  });

  it('should handle login failure', async () => {
    const { result } = renderHook(() => useAuth(), { wrapper });

    const mockError = new ApiError('Invalid credentials', 401);
    (api.login as vi.Mock).mockRejectedValue(mockError);

    await act(async () => {
      await expect(result.current.login('testuser', 'wrongpassword')).rejects.toThrow(
        'Invalid credentials',
      );
    });

    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.token).toBeNull();
    expect(localStorage.getItem('auth_token')).toBeNull();
  });

  it('should log out and clear state', async () => {
    // First, log in
    const { result } = renderHook(() => useAuth(), { wrapper });
    const mockToken = 'test_token';
    (api.login as vi.Mock).mockResolvedValue({ access_token: mockToken });
    await act(async () => {
      await result.current.login('testuser', 'password');
    });

    // Then, log out
    await act(async () => {
      result.current.logout();
    });

    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.token).toBeNull();
    expect(localStorage.getItem('auth_token')).toBeNull();
    expect(api.setAuthToken).toHaveBeenCalledWith(null);
    expect(mockedNavigate).toHaveBeenCalledWith('/login');
  });

  it('should handle unauthorized events', () => {
    const { result } = renderHook(() => useAuth(), { wrapper });

    act(() => {
      window.dispatchEvent(new CustomEvent('cortex-unauthorized'));
    });

    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.token).toBeNull();
    expect(mockedNavigate).toHaveBeenCalledWith('/login');
  });
});
