
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { api, request, ApiError } from './api';
import { logger } from './logger';

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
        url: 'http://localhost' + mockEndpoint,
        json: async () => errorDetails,
      });

      await expect(request(mockEndpoint)).rejects.toThrow(ApiError);
      try {
        await request(mockEndpoint);
      } catch (e) {
        const err = e as ApiError;
        expect(err.status).toBe(500);
        expect(err.details).toEqual(errorDetails);
        expect(err.message).toBe('API request failed with status 500');
      }
    });

    it('should throw ApiError for non-JSON error responses', async () => {
      (fetch as vi.Mock).mockResolvedValue({
        ok: false,
        status: 502,
        url: 'http://localhost' + mockEndpoint,
        statusText: 'Bad Gateway',
        json: async () => {
          throw new SyntaxError('Invalid JSON');
        },
      });

      try {
        await request(mockEndpoint);
      } catch (e) {
        const err = e as ApiError;
        expect(err.status).toBe(502);
        expect(err.details).toEqual({ detail: 'Bad Gateway' });
      }
    });

    it('should throw ApiError on JSON parsing failure for a 200 response', async () => {
      (fetch as vi.Mock).mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => {
          throw new SyntaxError('Invalid JSON');
        },
      });

      await expect(request(mockEndpoint)).rejects.toThrow(
        new ApiError('Failed to parse JSON response.', 200),
      );
    });

    it('should dispatch cortex-unauthorized event on 401 error', async () => {
      const dispatchEventSpy = vi.spyOn(window, 'dispatchEvent');
      (fetch as vi.Mock).mockResolvedValue({
        ok: false,
        status: 401,
        url: 'http://localhost' + mockEndpoint,
        json: async () => ({ detail: 'Unauthorized' }),
      });

      await expect(request(mockEndpoint)).rejects.toThrow(ApiError);
      expect(dispatchEventSpy).toHaveBeenCalledWith(new CustomEvent('cortex-unauthorized'));
    });

    it('should dispatch api:error event on non-401 errors', async () => {
      const dispatchEventSpy = vi.spyOn(window, 'dispatchEvent');
      const errorDetails = { detail: 'Server error' };
      (fetch as vi.Mock).mockResolvedValue({
        ok: false,
        status: 500,
        url: 'http://localhost' + mockEndpoint,
        json: async () => errorDetails,
      });

      await expect(request(mockEndpoint)).rejects.toThrow(ApiError);

      expect(dispatchEventSpy).toHaveBeenCalledWith(
        new CustomEvent('api:error', {
          detail: new ApiError('API request failed with status 500', 500, errorDetails),
        }),
      );
    });

    it('should not dispatch cortex-unauthorized event on 401 error for login endpoint', async () => {
      const dispatchEventSpy = vi.spyOn(window, 'dispatchEvent');
      (fetch as vi.Mock).mockResolvedValue({
        ok: false,
        status: 401,
        url: 'http://localhost/api/v1/auth/login',
        json: async () => ({ detail: 'Unauthorized' }),
      });

      await expect(request('/api/v1/auth/login', {}, false)).rejects.toThrow(ApiError);
      expect(dispatchEventSpy).not.toHaveBeenCalledWith(new CustomEvent('cortex-unauthorized'));
    });

    it('should throw a generic ApiError for network failures and log sanitized config', async () => {
      (fetch as vi.Mock).mockRejectedValue(new TypeError('Network failed'));

      await expect(request(mockEndpoint, { body: 'sensitive' })).rejects.toThrow(
        'A network error occurred. Please try again later.',
      );

      expect(logger.error).toHaveBeenCalledWith(
        'A network or other fetch error occurred.',
        expect.objectContaining({
          endpoint: mockEndpoint,
          config: expect.objectContaining({ body: '[REDACTED]' }),
          error: 'Network failed',
        }),
      );
    });

    it('should support AbortController signals', async () => {
      const controller = new AbortController();
      const signal = controller.signal;
      (fetch as vi.Mock).mockRejectedValue(new DOMException('Aborted', 'AbortError'));

      controller.abort();

      await expect(request(mockEndpoint, { signal })).rejects.toThrow(
        'A network error occurred. Please try again later.',
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

  describe('Authentication and Headers', () => {
    it('should include Authorization header if token exists', async () => {
      const token = 'test-token';
      api.setAuthToken(token);
      (fetch as vi.Mock).mockResolvedValue({ ok: true, json: async () => ({}) });

      await api.fetchStatus();

      const fetchCall = (fetch as vi.Mock).mock.calls[0];
      const headers = fetchCall[1].headers as Record<string, string>;
      expect(headers['Authorization']).toBe(`Bearer ${token}`);
    });

    it('should not include Authorization header if token does not exist', async () => {
      api.setAuthToken(null);
      (fetch as vi.Mock).mockResolvedValue({ ok: true, json: async () => ({}) });

      await api.fetchStatus();

      const fetchCall = (fetch as vi.Mock).mock.calls[0];
      const headers = fetchCall[1].headers as Record<string, string>;
      expect(headers['Authorization']).toBeUndefined();
    });

    it('should not include Authorization header for unauthenticated requests', async () => {
      api.setAuthToken('some-token');
      (fetch as vi.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({ access_token: 'new' }),
      });

      await api.fetchHealth(); // An unauthenticated endpoint

      const fetchCall = (fetch as vi.Mock).mock.calls[0];
      const headers = fetchCall[1].headers as Record<string, string>;
      expect(headers['Authorization']).toBeUndefined();
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

  // ===========================================================================
  // Specific endpoint tests
  // ===========================================================================

  describe('api.login', () => {
    it('should send credentials as x-www-form-urlencoded', async () => {
      (fetch as vi.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({ access_token: 'new-token' }),
      });

      await api.login('user', 'pass');

      expect(fetch).toHaveBeenCalledWith(
        '/api/v1/auth/login',
        expect.objectContaining({
          method: 'POST',
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
          },
        }),
      );

      // Verify the body is a URLSearchParams instance
      const fetchCall = (fetch as vi.Mock).mock.calls[0];
      const body = fetchCall[1].body;
      expect(body).toBeInstanceOf(URLSearchParams);
      expect(body.get('username')).toBe('user');
      expect(body.get('password')).toBe('pass');
    });

    it('should not send an Authorization header', async () => {
        api.setAuthToken('existing-token');
        (fetch as vi.Mock).mockResolvedValue({
          ok: true,
          json: async () => ({ access_token: 'new-token' }),
        });

        await api.login('user', 'pass');

        const fetchCall = (fetch as vi.Mock).mock.calls[0];
        const headers = fetchCall[1].headers as Record<string, string>;
        expect(headers['Authorization']).toBeUndefined();
      });
  });

  describe('api.runDoctor', () => {
    it('should make an authenticated POST request', async () => {
        api.setAuthToken('admin-token');
        (fetch as vi.Mock).mockResolvedValue({
          ok: true,
          json: async () => ({ overall_status: 'healthy' }),
        });

        await api.runDoctor();

        expect(fetch).toHaveBeenCalledWith(
          '/api/v1/admin/doctor',
          expect.objectContaining({
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer admin-token'
            }
          }),
        );
      });
  })
});
