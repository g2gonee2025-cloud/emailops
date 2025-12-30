import { render, screen, waitFor } from './testUtils';
import DashboardView from '../views/DashboardView';
import { vi } from 'vitest';
import { api } from '../lib/api';
import type { HealthResponse } from '../lib/api';

vi.mock('../lib/api');

const mockedApi = vi.mocked(api);

describe('DashboardView', () => {
  afterEach(() => {
    vi.resetAllMocks();
  });

  it('should display loading state initially', () => {
    // Mock a pending promise
    mockedApi.fetchHealth.mockReturnValue(new Promise(() => {}));
    render(<DashboardView />);
    expect(screen.getByText('CHECKING...')).toBeInTheDocument();
  });

  it('should display health status on successful fetch', async () => {
    const mockHealth: HealthResponse = {
      status: 'healthy',
      version: '1.2.3',
    };
    mockedApi.fetchHealth.mockResolvedValue(mockHealth);
    render(<DashboardView />);

    await waitFor(() => {
      expect(screen.getByText('HEALTHY')).toBeInTheDocument();
      expect(screen.getByText('v1.2.3')).toBeInTheDocument();
    });
  });

  it('should display loading state if fetch fails', async () => {
    mockedApi.fetchHealth.mockRejectedValue(new Error('API Error'));
    render(<DashboardView />);

    // The component doesn't handle errors, so it will remain in the "CHECKING..." state.
    // We wait for any potential state updates, then assert the loading state is still present.
    await waitFor(() => {
       expect(screen.getByText('CHECKING...')).toBeInTheDocument();
    });

    // Verify that the success state is not shown
    expect(screen.queryByText('HEALTHY')).not.toBeInTheDocument();
  });

  it('should render all KPI cards and other sections', () => {
    mockedApi.fetchHealth.mockResolvedValue({ status: 'healthy', version: '1.2.3' });
    render(<DashboardView />);

    expect(screen.getByText('Mission Control')).toBeInTheDocument();
    expect(screen.getByText('Pipeline Throughput')).toBeInTheDocument();
    expect(screen.getByText('Active Connections')).toBeInTheDocument();
    expect(screen.getByText('Vector Index')).toBeInTheDocument();
    expect(screen.getByText('Security Gate')).toBeInTheDocument();
    expect(screen.getByText('Live Process Stream')).toBeInTheDocument();
    expect(screen.getByText('System Uptime')).toBeInTheDocument();
    expect(screen.getByText('Email Processing')).toBeInTheDocument();
    expect(screen.getByText('RAG Performance')).toBeInTheDocument();
  });
});
