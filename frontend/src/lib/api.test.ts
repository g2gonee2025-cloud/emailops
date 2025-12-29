// frontend/src/lib/api.test.ts
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { api, ApiError } from './api';

// Mock the global fetch
global.fetch = vi.fn();

// Mock logger to avoid polluting test output
vi.mock('./logger', () => ({
  logger: {
    error: vi.fn(),
  },
}));

describe('API Client', () => {
  beforeEach(() => {
    // Set a default mock implementation for fetch
    vi.mocked(fetch).mockImplementation(async (url: RequestInfo | URL, options?: RequestInit) => {
      if (options?.signal?.aborted) {
        const error = new DOMException('The user aborted a request.', 'AbortError');
        throw error;
      }
      return Promise.resolve({
        ok: true,
        status: 200,
        statusText: 'OK',
        json: async () => ({ success: true }),
        text: async () => JSON.stringify({ success: true }),
      } as Response);
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    api.setAuthToken(null);
  });

  describe('request wrapper', () => {
    it('should make a successful GET request', async () => {
      const data = await api.fetchHealth();
      expect(fetch).toHaveBeenCalledWith('/health', expect.any(Object));
      expect(data).toEqual({ success: true });
    });

    it('should make a successful POST request with a body', async () => {
      await api.search('test query');
      expect(fetch).toHaveBeenCalledWith(
        '/api/v1/search',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ query: 'test query', k: 10, filters: {} }),
        })
      );
    });

    it('should include Authorization header when token is set', async () => {
      api.setAuthToken('test-token');
      await api.fetchStatus();
      expect(fetch).toHaveBeenCalledWith(
        '/api/v1/admin/status',
        expect.objectContaining({
          headers: expect.objectContaining({
            Authorization: 'Bearer test-token',
          }),
        })
      );
    });

    it('should handle API errors (e.g., 404 Not Found)', async () => {
      const errorResponse = {
        ok: false,
        status: 404,
        statusText: 'Not Found',
        json: async () => ({ detail: 'The requested resource was not found' }),
      };
      vi.mocked(fetch).mockResolvedValue(errorResponse as Response);

      await expect(api.fetchHealth()).rejects.toThrow(ApiError);
      await expect(api.fetchHealth()).rejects.toMatchObject({
        status: 404,
        message: 'The requested resource was not found',
      });
    });

    it('should handle network errors', async () => {
      vi.mocked(fetch).mockRejectedValue(new TypeError('Failed to fetch'));

      await expect(api.fetchHealth()).rejects.toThrow(ApiError);
      await expect(api.fetchHealth()).rejects.toMatchObject({
        status: 0,
        message: 'Failed to fetch',
      });
    });

    it('should use AbortController signal', async () => {
      const controller = new AbortController();
      controller.abort();

      await expect(api.fetchHealth(controller.signal)).rejects.toThrow(ApiError);
    });
  });
});
