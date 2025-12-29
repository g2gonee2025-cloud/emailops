
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { api, request, ApiError } from './api';

// Mock the logger to avoid polluting test output
vi.mock('./logger', () => ({
  logger: {
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
      (fetch as vi.Mock).mockResolvedValue({
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
      (fetch as vi.Mock).mockResolvedValue({
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
      (fetch as vi.Mock).mockResolvedValue({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        url: 'http://localhost/api/v1' + mockEndpoint,
        json: async () => errorDetails,
      });

      await expect(request(mockEndpoint)).rejects.toThrow(ApiError);
      try {
        await request(mockEndpoint);
      } catch (e) {
        const err = e as ApiError;
        expect(err.status).toBe(500);
        expect(err.detail).toEqual(errorDetails.detail);
        expect(err.message).toBe('API Error: 500 Internal Server Error');
      }
    });

    it('should dispatch cortex-unauthorized event on 401 error', async () => {
      const dispatchEventSpy = vi.spyOn(window, 'dispatchEvent');
      (fetch as vi.Mock).mockResolvedValue({
        ok: false,
        status: 401,
        statusText: 'Unauthorized',
        url: 'http://localhost/api/v1' + mockEndpoint,
        json: async () => ({ detail: 'Unauthorized' }),
      });

      await expect(request(mockEndpoint)).rejects.toThrow(ApiError);
      expect(dispatchEventSpy).toHaveBeenCalledWith(new CustomEvent('cortex-unauthorized'));
    });

    it('should not dispatch cortex-unauthorized event for login endpoint', async () => {
      const dispatchEventSpy = vi.spyOn(window, 'dispatchEvent');
      (fetch as vi.Mock).mockResolvedValue({
        ok: false,
        status: 401,
        statusText: 'Unauthorized',
        url: 'http://localhost/api/v1/auth/login',
        json: async () => ({ detail: 'Unauthorized' }),
      });

      await expect(request('/api/v1/auth/login', {}, false)).rejects.toThrow(ApiError);
      // It should dispatch api:error, but not cortex-unauthorized
      expect(dispatchEventSpy).not.toHaveBeenCalledWith(
        new CustomEvent('cortex-unauthorized'),
      );
    });

    it('should throw a specific ApiError for network failures', async () => {
      (fetch as vi.Mock).mockRejectedValue(new TypeError('Network failed'));
      await expect(request(mockEndpoint)).rejects.toThrow('API Error: 503 Service Unavailable');
    });

    it('should support AbortController signals and throw ApiError on abort', async () => {
      const controller = new AbortController();
      const signal = controller.signal;
      (fetch as vi.Mock).mockRejectedValue(new DOMException('Aborted', 'AbortError'));

      controller.abort();

      await expect(request(mockEndpoint, { signal })).rejects.toThrow(
        'API Error: 503 Service Unavailable',
      );
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
      api.setAuthToken(token); // Sets token in localStorage
      (fetch as vi.Mock).mockResolvedValue({ ok: true, json: async () => ({}) });

      await api.fetchStatus();

      const fetchCall = (fetch as vi.Mock).mock.calls[0];
      const headers = fetchCall[1].headers as Record<string, string>;
      expect(headers['Authorization']).toBe(`Bearer ${token}`);
    });

    it('should not include Authorization header if token does not exist', async () => {
      api.setAuthToken(null); // Removes token from localStorage
      (fetch as vi.Mock).mockResolvedValue({ ok: true, json: async () => ({}) });

      await api.fetchStatus();

      const fetchCall = (fetch as vi.Mock).mock.calls[0];
      const headers = fetchCall[1].headers as Record<string, string>;
      expect(headers['Authorization']).toBeUndefined();
    });

    it('should not include Authorization header when includeAuth is false', async () => {
      api.setAuthToken('some-token');
      (fetch as vi.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({ access_token: 'new' }),
      });

      // Simulate a login call where the auth header should not be included
      await request('/api/v1/auth/login', { method: 'POST' }, false);

      const fetchCall = (fetch as vi.Mock).mock.calls[0];
      const requestConfig = fetchCall[1] as RequestInit;
      expect(requestConfig.headers).not.toHaveProperty('Authorization');
      expect(requestConfig.headers).toHaveProperty('Content-Type');
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
