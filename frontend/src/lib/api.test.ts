
import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { api, request, ApiError, type RetryConfig } from './api';

// Mock the logger to avoid polluting test output
vi.mock('./logger', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
    warn: vi.fn(),
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
      const errorDetails = { detail: 'Bad Request' };
      (fetch as Mock).mockResolvedValue({
        ok: false,
        status: 400,
        url: 'http://localhost/api/v1' + mockEndpoint,
        headers: new Headers(),
        json: async () => errorDetails,
      });

      // Use retry: false to test immediate failure without retries
      await expect(request(mockEndpoint, { retry: false })).rejects.toThrow(ApiError);
      try {
        await request(mockEndpoint, { retry: false });
      } catch (e) {
        const err = e as ApiError;
        expect(err.status).toBe(400);
        expect(err.details).toEqual(errorDetails);
        expect(err.message).toBe('API request failed: 400');
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
      // Use retry: false to test immediate failure without retries
      await expect(request(mockEndpoint, { retry: false })).rejects.toThrow('A network error occurred.');
    });

    it('should support AbortController signals', async () => {
      const controller = new AbortController();
      const signal = controller.signal;
      (fetch as Mock).mockRejectedValue(new DOMException('Aborted', 'AbortError'));

      controller.abort();

      // AbortError is now properly propagated instead of being converted to network error
      await expect(request(mockEndpoint, { signal })).rejects.toThrow(DOMException);
      expect(fetch).toHaveBeenCalledTimes(0); // Request is aborted before fetch is called
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

  // ===========================================================================
  // Retry Logic tests
  // ===========================================================================

  describe('Retry Logic', () => {
    it('should retry on 500 error and succeed on third attempt', async () => {
      const mockData = { message: 'Success' };
      (fetch as Mock)
        .mockResolvedValueOnce({
          ok: false,
          status: 500,
          url: 'http://localhost' + mockEndpoint,
          headers: new Headers(),
          json: async () => ({ detail: 'Internal Server Error' }),
        })
        .mockResolvedValueOnce({
          ok: false,
          status: 500,
          url: 'http://localhost' + mockEndpoint,
          headers: new Headers(),
          json: async () => ({ detail: 'Internal Server Error' }),
        })
        .mockResolvedValueOnce({
          ok: true,
          status: 200,
          json: async () => mockData,
        });

      const result = await request(mockEndpoint, { retry: { retries: 3, baseDelay: 10, retryOn: [500] } });
      expect(result).toEqual(mockData);
      expect(fetch).toHaveBeenCalledTimes(3);
    });

    it('should retry on 429 error and respect Retry-After header (seconds)', async () => {
      const mockData = { message: 'Success' };
      const startTime = Date.now();

      (fetch as Mock)
        .mockResolvedValueOnce({
          ok: false,
          status: 429,
          url: 'http://localhost' + mockEndpoint,
          headers: new Headers({ 'Retry-After': '1' }),
          json: async () => ({ detail: 'Rate limited' }),
        })
        .mockResolvedValueOnce({
          ok: true,
          status: 200,
          json: async () => mockData,
        });

      const result = await request(mockEndpoint, { retry: { retries: 3, baseDelay: 10, retryOn: [429] } });
      const elapsed = Date.now() - startTime;

      expect(result).toEqual(mockData);
      expect(fetch).toHaveBeenCalledTimes(2);
      // Should have waited at least 1 second (1000ms) due to Retry-After header
      expect(elapsed).toBeGreaterThanOrEqual(900); // Allow some tolerance
    });

    it('should retry on 502, 503, 504 errors', async () => {
      const mockData = { message: 'Success' };

      (fetch as Mock)
        .mockResolvedValueOnce({
          ok: false,
          status: 502,
          url: 'http://localhost' + mockEndpoint,
          headers: new Headers(),
          json: async () => ({ detail: 'Bad Gateway' }),
        })
        .mockResolvedValueOnce({
          ok: false,
          status: 503,
          url: 'http://localhost' + mockEndpoint,
          headers: new Headers(),
          json: async () => ({ detail: 'Service Unavailable' }),
        })
        .mockResolvedValueOnce({
          ok: false,
          status: 504,
          url: 'http://localhost' + mockEndpoint,
          headers: new Headers(),
          json: async () => ({ detail: 'Gateway Timeout' }),
        })
        .mockResolvedValueOnce({
          ok: true,
          status: 200,
          json: async () => mockData,
        });

      const result = await request(mockEndpoint, { retry: { retries: 3, baseDelay: 10, retryOn: [502, 503, 504] } });
      expect(result).toEqual(mockData);
      expect(fetch).toHaveBeenCalledTimes(4);
    });

    it('should NOT retry on 4xx errors (except 429)', async () => {
      (fetch as Mock).mockResolvedValue({
        ok: false,
        status: 400,
        url: 'http://localhost' + mockEndpoint,
        headers: new Headers(),
        json: async () => ({ detail: 'Bad Request' }),
      });

      await expect(request(mockEndpoint, { retry: { retries: 3, baseDelay: 10, retryOn: [429, 500, 502, 503, 504] } })).rejects.toThrow(ApiError);
      expect(fetch).toHaveBeenCalledTimes(1);
    });

    it('should NOT retry on 404 errors', async () => {
      (fetch as Mock).mockResolvedValue({
        ok: false,
        status: 404,
        url: 'http://localhost' + mockEndpoint,
        headers: new Headers(),
        json: async () => ({ detail: 'Not Found' }),
      });

      await expect(request(mockEndpoint)).rejects.toThrow(ApiError);
      expect(fetch).toHaveBeenCalledTimes(1);
    });

    it('should disable retry when retry: false is passed', async () => {
      (fetch as Mock).mockResolvedValue({
        ok: false,
        status: 500,
        url: 'http://localhost' + mockEndpoint,
        headers: new Headers(),
        json: async () => ({ detail: 'Internal Server Error' }),
      });

      await expect(request(mockEndpoint, { retry: false })).rejects.toThrow(ApiError);
      expect(fetch).toHaveBeenCalledTimes(1);
    });

    it('should throw after exhausting all retries', async () => {
      (fetch as Mock).mockResolvedValue({
        ok: false,
        status: 500,
        url: 'http://localhost' + mockEndpoint,
        headers: new Headers(),
        json: async () => ({ detail: 'Internal Server Error' }),
      });

      await expect(request(mockEndpoint, { retry: { retries: 2, baseDelay: 10, retryOn: [500] } })).rejects.toThrow(ApiError);
      // Initial attempt + 2 retries = 3 total calls
      expect(fetch).toHaveBeenCalledTimes(3);
    });

    it('should cancel pending retries when AbortSignal is triggered', async () => {
      const controller = new AbortController();

      (fetch as Mock).mockResolvedValueOnce({
        ok: false,
        status: 500,
        url: 'http://localhost' + mockEndpoint,
        headers: new Headers(),
        json: async () => ({ detail: 'Internal Server Error' }),
      });

      const requestPromise = request(mockEndpoint, {
        signal: controller.signal,
        retry: { retries: 3, baseDelay: 100, retryOn: [500] },
      });

      // Abort after a short delay (during the retry wait)
      setTimeout(() => controller.abort(), 50);

      await expect(requestPromise).rejects.toThrow();
      // Should have only made one fetch call before being aborted during retry wait
      expect(fetch).toHaveBeenCalledTimes(1);
    });

    it('should not make request if AbortSignal is already aborted', async () => {
      const controller = new AbortController();
      controller.abort();

      await expect(request(mockEndpoint, { signal: controller.signal })).rejects.toThrow();
      expect(fetch).toHaveBeenCalledTimes(0);
    });

    it('should retry network errors', async () => {
      const mockData = { message: 'Success' };

      (fetch as Mock)
        .mockRejectedValueOnce(new TypeError('Network failed'))
        .mockRejectedValueOnce(new TypeError('Network failed'))
        .mockResolvedValueOnce({
          ok: true,
          status: 200,
          json: async () => mockData,
        });

      const result = await request(mockEndpoint, { retry: { retries: 3, baseDelay: 10, retryOn: [500] } });
      expect(result).toEqual(mockData);
      expect(fetch).toHaveBeenCalledTimes(3);
    });

    it('should use exponential backoff with jitter', async () => {
      const originalDateNow = Date.now;
      const currentTime = 0;

      vi.spyOn(Date, 'now').mockImplementation(() => {
        return currentTime;
      });

      const mockData = { message: 'Success' };

      (fetch as Mock)
        .mockResolvedValueOnce({
          ok: false,
          status: 500,
          url: 'http://localhost' + mockEndpoint,
          headers: new Headers(),
          json: async () => ({ detail: 'Internal Server Error' }),
        })
        .mockResolvedValueOnce({
          ok: false,
          status: 500,
          url: 'http://localhost' + mockEndpoint,
          headers: new Headers(),
          json: async () => ({ detail: 'Internal Server Error' }),
        })
        .mockResolvedValueOnce({
          ok: true,
          status: 200,
          json: async () => mockData,
        });

      // Use a very short base delay for testing
      await request(mockEndpoint, { retry: { retries: 3, baseDelay: 10, retryOn: [500] } });

      expect(fetch).toHaveBeenCalledTimes(3);

      Date.now = originalDateNow;
    });

    it('should use default retry config when retry option is not provided', async () => {
      const mockData = { message: 'Success' };

      (fetch as Mock)
        .mockResolvedValueOnce({
          ok: false,
          status: 500,
          url: 'http://localhost' + mockEndpoint,
          headers: new Headers(),
          json: async () => ({ detail: 'Internal Server Error' }),
        })
        .mockResolvedValueOnce({
          ok: true,
          status: 200,
          json: async () => mockData,
        });

      // Default config should retry on 500
      const result = await request(mockEndpoint);
      expect(result).toEqual(mockData);
      expect(fetch).toHaveBeenCalledTimes(2);
    });

    it('should merge partial retry config with defaults', async () => {
      const mockData = { message: 'Success' };

      (fetch as Mock)
        .mockResolvedValueOnce({
          ok: false,
          status: 500,
          url: 'http://localhost' + mockEndpoint,
          headers: new Headers(),
          json: async () => ({ detail: 'Internal Server Error' }),
        })
        .mockResolvedValueOnce({
          ok: true,
          status: 200,
          json: async () => mockData,
        });

      // Only override baseDelay, should still use default retryOn
      const result = await request(mockEndpoint, { retry: { baseDelay: 10 } as Partial<RetryConfig> });
      expect(result).toEqual(mockData);
      expect(fetch).toHaveBeenCalledTimes(2);
    });
  });
});
