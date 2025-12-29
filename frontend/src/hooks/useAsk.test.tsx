/** @vitest-environment jsdom */
import { describe, it, expect, vi } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { useAsk } from './useAsk';
import { api } from '@/lib/api';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import React from 'react';

// Mock the api module
vi.mock('@/lib/api', () => ({
  api: {
    ask: vi.fn(),
  },
}));

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe('useAsk', () => {
  it('should return initial state correctly', () => {
    const { result } = renderHook(() => useAsk(), { wrapper: createWrapper() });

    expect(result.current.data).toBeUndefined();
    expect(result.current.isLoading).toBe(false);
    expect(result.current.error).toBe(null);
  });

  it('should call api.ask with the correct variables', async () => {
    const { result } = renderHook(() => useAsk(), { wrapper: createWrapper() });
    const testVariables = { query: 'test query', threadId: '123', k: 5 };

    act(() => {
      result.current.ask(testVariables);
    });

    await waitFor(() => {
        expect(api.ask).toHaveBeenCalledWith(testVariables.query, testVariables.threadId, testVariables.k);
    });
  });

  it('should update state on successful mutation', async () => {
    const mockData = {
      answer: { text: 'test answer', confidence_overall: 0.9 },
      confidence: 0.9,
    };
    (api.ask as vi.Mock).mockResolvedValue(mockData);

    const { result } = renderHook(() => useAsk(), { wrapper: createWrapper() });

    act(() => {
      result.current.ask({ query: 'test' });
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toEqual(mockData);
    expect(result.current.error).toBe(null);
  });

  it('should update state on failed mutation', async () => {
    const mockError = new Error('test error');
    (api.ask as vi.Mock).mockRejectedValue(mockError);

    const { result } = renderHook(() => useAsk(), { wrapper: createWrapper() });

    act(() => {
      result.current.ask({ query: 'test' });
    });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.error).toEqual(mockError);
  });
});
