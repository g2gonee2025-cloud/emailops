/** @vitest-environment jsdom */

import { describe, it, expect, vi } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useDraft } from './useDraft';
import { api } from '@/lib/api';
import React from 'react';

// Mock the api module
vi.mock('@/lib/api', () => ({
  api: {
    draftEmail: vi.fn(),
  },
}));

const queryClient = new QueryClient();
const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
);

describe('useDraft', () => {
  it('should call api.draftEmail and return the draft', async () => {
    const mockDraftResponse = {
      draft: {
        subject: 'Test Subject',
        body: 'Test Body',
      },
      confidence: 0.9,
      iterations: 1,
    };
    (api.draftEmail as vi.Mock).mockResolvedValue(mockDraftResponse);

    const { result } = renderHook(() => useDraft(), { wrapper });

    result.current.mutate({
      instruction: 'Test instruction',
      threadId: 'test-thread-id',
      tone: 'professional',
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(api.draftEmail).toHaveBeenCalledWith(
      'Test instruction',
      'test-thread-id',
      'professional',
    );
    expect(result.current.data).toEqual(mockDraftResponse);
  });
});
