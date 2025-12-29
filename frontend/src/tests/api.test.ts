
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { api, SearchResponse } from '../lib/api';
import { logger } from '../lib/logger';

describe('api client', () => {
  beforeEach(() => {
    // Reset mocks and authToken before each test
    vi.resetAllMocks();
    api.setAuthToken(null);
  });

  // ---------------------------------------------------------------------------
  // Success Scenarios
  // ---------------------------------------------------------------------------

  it('should fetch health status successfully', async () => {
    const mockHealth = { status: 'ok' };
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockHealth),
    });

    const health = await api.fetchHealth();
    expect(health).toEqual(mockHealth);
    expect(global.fetch).toHaveBeenCalledWith('/health');
  });

  it('should handle successful search requests', async () => {
    const mockSearchResponse: SearchResponse = {
      results: [{ chunk_id: '1', conversation_id: 'c1', content: 'test', score: 0.9 }],
      total_count: 1,
      query_time_ms: 100,
    };
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockSearchResponse),
    });

    const response = await api.search('test query');
    expect(response).toEqual(mockSearchResponse);
    expect(global.fetch).toHaveBeenCalledWith('/api/v1/search', expect.any(Object));
  });

  // ---------------------------------------------------------------------------
  // Auth Header Scenarios
  // ---------------------------------------------------------------------------

  it('should not include Authorization header when token is not set', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ results: [] }),
    });

    await api.search('test');
    const fetchOptions = (global.fetch as vi.Mock).mock.calls[0][1];
    expect(fetchOptions.headers).not.toHaveProperty('Authorization');
  });

  it('should include Authorization header when token is set', async () => {
    const token = 'test-auth-token';
    api.setAuthToken(token);

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ results: [] }),
    });

    await api.search('test');
    const fetchOptions = (global.fetch as vi.Mock).mock.calls[0][1];
    expect(fetchOptions.headers).toHaveProperty('Authorization', `Bearer ${token}`);
  });

  // ---------------------------------------------------------------------------
  // Error Handling Scenarios
  // ---------------------------------------------------------------------------

  it('should throw a typed error for non-ok responses', async () => {
    const errorDetail = 'Internal Server Error';
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      json: () => Promise.resolve({ detail: errorDetail }),
    });

    await expect(api.search('test')).rejects.toThrow(errorDetail);
  });

  it('should throw a generic error if error payload is malformed', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      json: () => Promise.reject(new Error('Malformed JSON')), // Simulate JSON parsing error
    });

    await expect(api.search('test')).rejects.toThrow('Unknown error');
  });

  it('should dispatch an "unauthorized" event on 401 status', async () => {
    const dispatchEventSpy = vi.spyOn(window, 'dispatchEvent');
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 401,
      json: () => Promise.resolve({ detail: 'Unauthorized' }),
    });

    await expect(api.search('test')).rejects.toThrow('Unauthorized');
    expect(dispatchEventSpy).toHaveBeenCalledWith(new Event('unauthorized'));

    dispatchEventSpy.mockRestore();
  });

  it('should return null and log error for health check failure', async () => {
    const loggerErrorSpy = vi.spyOn(logger, 'error').mockImplementation(() => {});
    global.fetch = vi.fn().mockRejectedValue(new Error('Network failure'));

    const result = await api.fetchHealth();
    expect(result).toBeNull();
    expect(loggerErrorSpy).toHaveBeenCalled();

    loggerErrorSpy.mockRestore();
  });
});
