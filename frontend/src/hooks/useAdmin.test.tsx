/** @vitest-environment jsdom */
import { renderHook, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import React from 'react';
import { useAdmin } from './useAdmin';
import { api } from '@/lib/api';

// Mock the api module
vi.mock('@/lib/api', () => ({
  api: {
    fetchStatus: vi.fn(),
    fetchConfig: vi.fn(),
    runDoctor: vi.fn(),
  },
}));

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false, // Disable retries for tests
      },
    },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe('useAdmin', () => {
  it('should fetch status and config on mount', async () => {
    const mockStatus = { status: 'healthy' };
    const mockConfig = { environment: 'development' };

    vi.mocked(api.fetchStatus).mockResolvedValue(mockStatus as any);
    vi.mocked(api.fetchConfig).mockResolvedValue(mockConfig as any);

    const { result } = renderHook(() => useAdmin(), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isStatusLoading).toBe(false);
      expect(result.current.isConfigLoading).toBe(false);
    });

    expect(result.current.status).toEqual(mockStatus);
    expect(result.current.config).toEqual(mockConfig);
    expect(api.fetchStatus).toHaveBeenCalledTimes(1);
    expect(api.fetchConfig).toHaveBeenCalledTimes(1);
  });

  it('should handle errors when fetching status and config', async () => {
    const statusError = new Error('Failed to fetch status');
    const configError = new Error('Failed to fetch config');

    vi.mocked(api.fetchStatus).mockRejectedValue(statusError);
    vi.mocked(api.fetchConfig).mockRejectedValue(configError);

    const { result } = renderHook(() => useAdmin(), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isStatusLoading).toBe(false);
      expect(result.current.isConfigLoading).toBe(false);
    });

    expect(result.current.statusError).toBe(statusError);
    expect(result.current.configError).toBe(configError);
  });

  it('should call runDoctor mutation', async () => {
    const mockDoctorReport = { overall_status: 'healthy', checks: [] };
    vi.mocked(api.runDoctor).mockResolvedValue(mockDoctorReport as any);

    const { result } = renderHook(() => useAdmin(), { wrapper: createWrapper() });

    result.current.runDoctor();

    await waitFor(() => {
        expect(result.current.isDoctorRunning).toBe(false);
    });

    expect(api.runDoctor).toHaveBeenCalledTimes(1);
    expect(result.current.doctorReport).toEqual(mockDoctorReport);
  });

  it('should handle error when running doctor', async () => {
    const doctorError = new Error('Failed to run doctor');
    vi.mocked(api.runDoctor).mockRejectedValue(doctorError);

    const { result } = renderHook(() => useAdmin(), { wrapper: createWrapper() });

    result.current.runDoctor();

    await waitFor(() => {
        expect(result.current.isDoctorRunning).toBe(false);
    });

    expect(result.current.doctorError).toBe(doctorError);
  });
});
