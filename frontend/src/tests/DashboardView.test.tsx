import { render, screen, waitFor } from './testUtils';
import DashboardView from '../views/DashboardView';
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { api } from '../lib/api';
import type { HealthResponse } from '../lib/api';

vi.mock('../lib/api');

const mockedApi = vi.mocked(api);

describe('DashboardView', () => {
  beforeEach(() => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.clearAllMocks();
  });

  it('should display loading state initially', () => {
    mockedApi.fetchHealth.mockReturnValue(new Promise(() => {}));
    render(<DashboardView />);
    expect(screen.getByText('CHECKING...')).toBeInTheDocument();
  });

  it('should display health status on successful fetch', async () => {
    const mockHealth: HealthResponse = {
      status: 'healthy',
      version: '1.2.3',
      environment: 'test',
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

    await waitFor(() => {
       expect(screen.getByText('CHECKING...')).toBeInTheDocument();
    });

    expect(screen.queryByText('HEALTHY')).not.toBeInTheDocument();
  });

  it('should render all dashboard sections', async () => {
    mockedApi.fetchHealth.mockResolvedValue({ status: 'healthy', version: '1.2.3', environment: 'test' });
    render(<DashboardView />);

    expect(screen.getByText('Mission Control')).toBeInTheDocument();
    expect(screen.getByText('Live Process Stream')).toBeInTheDocument();
    expect(screen.getByText('System Uptime')).toBeInTheDocument();
    expect(screen.getByText('Email Processing')).toBeInTheDocument();
    expect(screen.getByText('RAG Performance')).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('No KPI Data Available')).toBeInTheDocument();
    });
  });
});
