import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { api, ApiError } from '../lib/api';
import type { DraftEmailResponse } from '../lib/api';
import { useDraftEmail } from './useDraft';
import React from 'react';

// Mock the api module
vi.mock('../lib/api');

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
    },
  },
});

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
);

describe('useDraftEmail', () => {
  beforeEach(() => {
    vi.resetAllMocks();
    queryClient.clear();
  });

  it('should call api.draftEmail and return data on successful mutation', async () => {
    const mockResponse: DraftEmailResponse = {
      correlation_id: 'test-id',
      draft: {
        subject: 'Test Subject',
        body: 'Test Body',
        to: [],
        cc: [],
      },
      confidence: 0.9,
      iterations: 1,
    };

    const draftEmailSpy = vi.spyOn(api, 'draftEmail').mockResolvedValue(mockResponse);

    const { result } = renderHook(() => useDraftEmail(), { wrapper });

    const draftVariables = {
      instruction: 'Write a test email',
      threadId: 'thread-123',
      tone: 'casual',
    };

    result.current.mutate(draftVariables);

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(draftEmailSpy).toHaveBeenCalledWith(
      draftVariables.instruction,
      draftVariables.threadId,
      draftVariables.tone,
    );
    expect(result.current.data).toEqual(mockResponse);
    expect(result.current.error).toBeNull();
  });

  it('should return an error on a failed mutation', async () => {
    const mockError = new ApiError('Failed to draft email', 500);

    const draftEmailSpy = vi.spyOn(api, 'draftEmail').mockRejectedValue(mockError);

    const { result } = renderHook(() => useDraftEmail(), { wrapper });

    const draftVariables = {
      instruction: 'Write a failing email',
    };

    result.current.mutate(draftVariables);

    await waitFor(() => expect(result.current.isError).toBe(true));

    expect(draftEmailSpy).toHaveBeenCalledWith(
      draftVariables.instruction,
      undefined, // threadId
      undefined, // tone
    );
    expect(result.current.error).toBe(mockError);
    expect(result.current.data).toBeUndefined();
  });
});
