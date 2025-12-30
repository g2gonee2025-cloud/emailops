import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { api } from '../lib/api';
import { useSearch } from './useSearch';
import React from 'react';
import { vi } from 'vitest';

// Mock the API module
vi.mock('../lib/api', () => ({
  api: {
    search: vi.fn(),
  },
}));

const mockedApi = api as jest.Mocked<typeof api>;

const createTestQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={createTestQueryClient()}>{children}</QueryClientProvider>
);

describe('useSearch', () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it('should not call the API if the query is empty', () => {
    renderHook(() => useSearch({ query: '' }), { wrapper });
    expect(mockedApi.search).not.toHaveBeenCalled();
  });

  it('should call the API with the correct parameters when the query is provided', async () => {
    const mockResponse = { results: [], total_count: 0, query_time_ms: 10 };
    mockedApi.search.mockResolvedValue(mockResponse);

    const { result } = renderHook(() => useSearch({ query: 'test', k: 5 }), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(mockedApi.search).toHaveBeenCalledWith('test', 5, {});
  });

  it('should return data on successful API call', async () => {
    const mockResponse = {
      results: [{ chunk_id: '1', conversation_id: 'conv1', content: 'Test content', score: 0.9 }],
      total_count: 1,
      query_time_ms: 20,
    };
    mockedApi.search.mockResolvedValue(mockResponse);

    const { result } = renderHook(() => useSearch({ query: 'test' }), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.data).toEqual(mockResponse);
    expect(result.current.error).toBe(null);
    expect(result.current.isError).toBe(false);
  });

  it('should return an error when the API call fails', async () => {
    const mockError = new Error('API Error');
    mockedApi.search.mockRejectedValue(mockError);

    const { result } = renderHook(() => useSearch({ query: 'test' }), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.error).toBe(mockError);
    expect(result.current.isError).toBe(true);
    expect(result.current.data).toBe(undefined);
  });
});
