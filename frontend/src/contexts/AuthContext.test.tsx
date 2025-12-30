import { renderHook, act } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { AuthProvider, useAuth } from './AuthContext';
import { vi } from 'vitest';
import { api } from '@/lib/api';

// Mock the api module
vi.mock('@/lib/api', () => ({
  api: {
    setAuthToken: vi.fn(),
  },
}));

// Mock useNavigate
const mockedNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...(actual as object),
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

  it('should update state when setToken is called', () => {
    const { result } = renderHook(() => useAuth(), { wrapper });
    const mockToken = 'test_token';

    act(() => {
      result.current.setToken(mockToken);
    });

    expect(result.current.isAuthenticated).toBe(true);
    expect(result.current.token).toBe(mockToken);
    expect(localStorage.getItem('auth_token')).toBe(mockToken);
    expect(api.setAuthToken).toHaveBeenCalledWith(mockToken);
  });

  it('should log out and clear state', () => {
    const { result } = renderHook(() => useAuth(), { wrapper });
    const mockToken = 'test_token';

    // Set initial logged-in state
    act(() => {
      result.current.setToken(mockToken);
    });
    expect(result.current.isAuthenticated).toBe(true); // Verify logged-in state

    // Then, log out
    act(() => {
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
