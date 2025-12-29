
import { renderHook, act } from '@testing-library/react';
import { AuthProvider, useAuth } from './AuthContext';
import { vi } from 'vitest';
import * as api from '../lib/api';
import { ApiError } from '../lib/api';
import { BrowserRouter } from 'react-router-dom';

// Mock the navigate function
const mockedNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
    const original = await vi.importActual('react-router-dom');
    return {
        ...original,
        useNavigate: () => mockedNavigate,
    };
});


describe('AuthProvider', () => {
  afterEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
  });

  it('should handle successful login', async () => {
    const requestSpy = vi.spyOn(api, 'request').mockResolvedValue({ access_token: 'test_token' });

    const wrapper = ({ children }) => <BrowserRouter><AuthProvider>{children}</AuthProvider></BrowserRouter>;
    const { result } = renderHook(() => useAuth(), { wrapper });

    await act(async () => {
      await result.current.login('test', 'password');
    });

    expect(result.current.isAuthenticated).toBe(true);
    expect(result.current.token).toBe('test_token');
    expect(localStorage.getItem('auth_token')).toBe('test_token');

    // Check the call arguments
    expect(requestSpy).toHaveBeenCalledTimes(1);
    const [endpoint, options] = requestSpy.mock.calls[0];
    expect(endpoint).toBe('/api/v1/auth/login');
    expect(options.method).toBe('POST');
    expect(options.headers).toEqual({ 'Content-Type': 'application/x-www-form-urlencoded' });
    expect(options.body).toBeInstanceOf(URLSearchParams);
    expect((options.body as URLSearchParams).get('username')).toBe('test');
    expect((options.body as URLSearchParams).get('password')).toBe('password');
  });

  it('should handle failed login', async () => {
    const requestSpy = vi.spyOn(api, 'request').mockRejectedValue(new ApiError(401, 'Unauthorized'));

    const wrapper = ({ children }) => <BrowserRouter><AuthProvider>{children}</AuthProvider></BrowserRouter>;
    const { result } = renderHook(() => useAuth(), { wrapper });

    await act(async () => {
        try {
            await result.current.login('test', 'wrong_password');
        } catch (error) {
            expect(error.message).toBe('Invalid credentials');
        }
    });

    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.token).toBe(null);
    expect(localStorage.getItem('auth_token')).toBe(null);
    expect(requestSpy).toHaveBeenCalledTimes(1);
  });

  it('should handle logout', async () => {
    localStorage.setItem('auth_token', 'test_token');
    const wrapper = ({ children }) => <BrowserRouter><AuthProvider>{children}</AuthProvider></BrowserRouter>;
    const { result } = renderHook(() => useAuth(), { wrapper });

    expect(result.current.isAuthenticated).toBe(true);

    act(() => {
        result.current.logout();
    });

    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.token).toBe(null);
    expect(localStorage.getItem('auth_token')).toBe(null);
    expect(mockedNavigate).toHaveBeenCalledWith('/login');
    });
});
