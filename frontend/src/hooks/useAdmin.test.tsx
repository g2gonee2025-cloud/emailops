import { renderHook, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { AllTheProviders } from '../tests/testUtils';
import { useAdmin } from './useAdmin';
import { api } from '../lib/api';

// Mock the api module
vi.mock('../lib/api');

describe('useAdmin', () => {
  it('should fetch system status', async () => {
    const mockStatus = { status: 'ok', service: 'test', env: 'dev' };
    vi.mocked(api.fetchStatus).mockResolvedValue(mockStatus);

    const { result } = renderHook(() => useAdmin(), { wrapper: AllTheProviders });

    await waitFor(() => expect(result.current.isStatusLoading).toBe(false));

    expect(result.current.status).toEqual(mockStatus);
  });

  it('should fetch system config', async () => {
    const mockConfig = { environment: 'dev', provider: 'test', log_level: 'info' };
    vi.mocked(api.fetchConfig).mockResolvedValue(mockConfig);

    const { result } = renderHook(() => useAdmin(), { wrapper: AllTheProviders });

    await waitFor(() => expect(result.current.isConfigLoading).toBe(false));

    expect(result.current.config).toEqual(mockConfig);
  });

  it('should run doctor', async () => {
    const mockReport = { overall_status: 'healthy' as const, checks: [] };
    vi.mocked(api.runDoctor).mockResolvedValue(mockReport);

    const { result } = renderHook(() => useAdmin(), { wrapper: AllTheProviders });

    result.current.runDoctor();

    await waitFor(() => expect(result.current.isDoctorRunning).toBe(false));

    expect(result.current.doctorReport).toEqual(mockReport);
  });
});
