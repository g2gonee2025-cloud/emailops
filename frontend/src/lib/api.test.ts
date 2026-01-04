
import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { api, request, ApiError } from './api';

// Mock the logger to avoid polluting test output
vi.mock('./logger', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

describe('API Client', () => {
  const mockEndpoint = '/test-endpoint';

  beforeEach(() => {
    // Reset fetch mock before each test
    global.fetch = vi.fn();
    // Clear localStorage
    localStorage.clear();
    // Reset spies
    vi.restoreAllMocks();
  });

  // ===========================================================================
  // request<T> wrapper tests
  // ===========================================================================

  describe('request<T>', () => {
    it('should return JSON data on a successful request', async () => {
      const mockData = { message: 'Success' };
      (fetch as Mock).mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => mockData,
      });

      const result = await request(mockEndpoint);
      expect(result).toEqual(mockData);
      expect(fetch).toHaveBeenCalledWith(mockEndpoint, {
        headers: {
          'Content-Type': 'application/json',
        },
      });
    });

    it('should handle 204 No Content responses', async () => {
      (fetch as Mock).mockResolvedValue({
        ok: true,
        status: 204,
        json: async () => {
          throw new Error('Should not be called');
        },
      });

      const result = await request(mockEndpoint);
      expect(result).toEqual({});
    });

    it('should throw ApiError for non-ok responses with JSON details', async () => {
      const errorDetails = { detail: 'Server error' };
      (fetch as Mock).mockResolvedValue({
        ok: false,
        status: 500,
        url: 'http://localhost/api/v1' + mockEndpoint,
        json: async () => errorDetails,
      });

      await expect(request(mockEndpoint)).rejects.toThrow(ApiError);
      try {
        await request(mockEndpoint);
      } catch (e) {
        const err = e as ApiError;
        expect(err.status).toBe(500);
        expect(err.details).toEqual(errorDetails);
        expect(err.message).toBe('API request failed: 500');
      }
    });

    it('should dispatch cortex-unauthorized event on 401 error', async () => {
      const dispatchEventSpy = vi.spyOn(window, 'dispatchEvent');
      (fetch as Mock).mockResolvedValue({
        ok: false,
        status: 401,
        url: 'http://localhost/api/v1' + mockEndpoint,
        json: async () => ({ detail: 'Unauthorized' }),
      });

      await expect(request(mockEndpoint)).rejects.toThrow(ApiError);
      expect(dispatchEventSpy).toHaveBeenCalledWith(new CustomEvent('cortex-unauthorized'));
    });

    it('should not dispatch cortex-unauthorized event on 401 error for login endpoint', async () => {
        const dispatchEventSpy = vi.spyOn(window, 'dispatchEvent');
        (fetch as Mock).mockResolvedValue({
          ok: false,
          status: 401,
          url: 'http://localhost/api/v1/auth/login',
          json: async () => ({ detail: 'Unauthorized' }),
        });

        await expect(request('/api/v1/auth/login', {}, false)).rejects.toThrow(ApiError);
        expect(dispatchEventSpy).not.toHaveBeenCalled();
      });

    it('should throw a generic error for network failures', async () => {
      (fetch as Mock).mockRejectedValue(new TypeError('Network failed'));
      await expect(request(mockEndpoint)).rejects.toThrow('A network error occurred.');
    });

    it('should support AbortController signals', async () => {
      const controller = new AbortController();
      const signal = controller.signal;
      (fetch as Mock).mockRejectedValue(new DOMException('Aborted', 'AbortError'));

      controller.abort();

      await expect(request(mockEndpoint, { signal })).rejects.toThrow('A network error occurred.');
      expect(fetch).toHaveBeenCalledWith(
        mockEndpoint,
        expect.objectContaining({
          signal,
        }),
      );
    });
  });

  // ===========================================================================
  // Header and Auth tests
  // ===========================================================================

  describe('Authentication', () => {
    it('should include Authorization header if token exists', async () => {
      const token = 'test-token';
      localStorage.setItem('auth_token', token); // Sets token in localStorage
      (global.fetch as Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: 'ok', env: 'test', service: 'cortex', version: '1.0.0' }),
      });

      await api.fetchStatus();

      const fetchCall = (fetch as Mock).mock.calls[0];
      const headers = fetchCall[1].headers as Record<string, string>;
      expect(headers['Authorization']).toBe('Bearer test-token');
    });

    it('should not include Authorization header if token does not exist', async () => {
      localStorage.removeItem('auth_token'); // Removes token from localStorage
      (global.fetch as Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: 'ok', env: 'test', service: 'cortex', version: '1.0.0' }),
      });

      await api.fetchStatus();

      const fetchCall = (fetch as Mock).mock.calls[0];
      const headers = fetchCall[1].headers as Record<string, string>;
      expect(headers['Authorization']).toBeUndefined();
    });

    it('should not include Authorization header for unauthenticated requests like login', async () => {
      api.setAuthToken('some-token');
      (fetch as Mock).mockResolvedValue({ ok: true, json: async () => ({ access_token: 'new' }) });

      await api.login('user', 'pass');

      const fetchCall = (fetch as Mock).mock.calls[0];
      const requestConfig = fetchCall[1] as RequestInit;
      expect(requestConfig.headers).not.toHaveProperty('Authorization');
    });

    it('setAuthToken should store token in localStorage', () => {
        const setItemSpy = vi.spyOn(Storage.prototype, 'setItem');
        const token = 'my-secret-token';
        api.setAuthToken(token);
        expect(setItemSpy).toHaveBeenCalledWith('auth_token', token);
      });

      it('setAuthToken should remove token from localStorage if token is null', () => {
        const removeItemSpy = vi.spyOn(Storage.prototype, 'removeItem');
        api.setAuthToken(null);
        expect(removeItemSpy).toHaveBeenCalledWith('auth_token');
      });
  });
});
