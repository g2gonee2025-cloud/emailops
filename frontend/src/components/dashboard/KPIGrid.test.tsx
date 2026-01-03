import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import KPIGrid, { type KPIData } from './KPIGrid';
import { AllTheProviders } from '../../tests/testUtils';

const testKpis: KPIData[] = [
  { id: 'total-emails', title: 'Total Emails Processed', value: '1,234,567', change: '+5.2%' },
  { id: 'avg-response-time', title: 'Average Response Time', value: '2.3 hours', change: '-1.5%' },
  { id: 'open-rate', title: 'Open Rate', value: '25.8%', change: '+0.8%' },
  { id: 'click-through-rate', title: 'Click-Through Rate', value: '4.2%', change: '+0.2%' },
];

describe('KPIGrid', () => {
  it('renders the KPI cards when data is provided', () => {
    render(
      <AllTheProviders>
        <KPIGrid kpis={testKpis} />
      </AllTheProviders>
    );

    expect(screen.getByText('Total Emails Processed')).toBeInTheDocument();
    expect(screen.getByText('Average Response Time')).toBeInTheDocument();
    expect(screen.getByText('Open Rate')).toBeInTheDocument();
    expect(screen.getByText('Click-Through Rate')).toBeInTheDocument();

    expect(screen.getByText('1,234,567')).toBeInTheDocument();
    expect(screen.getByText('2.3 hours')).toBeInTheDocument();
    expect(screen.getByText('25.8%')).toBeInTheDocument();
    expect(screen.getByText('4.2%')).toBeInTheDocument();

    expect(screen.getByText('+5.2%')).toBeInTheDocument();
    expect(screen.getByText('-1.5%')).toBeInTheDocument();
    expect(screen.getByText('+0.8%')).toBeInTheDocument();
    expect(screen.getByText('+0.2%')).toBeInTheDocument();
  });

  it('renders empty state when no KPIs are provided', () => {
    render(
      <AllTheProviders>
        <KPIGrid />
      </AllTheProviders>
    );

    expect(screen.getByText('No KPI Data Available')).toBeInTheDocument();
    expect(screen.getByText('KPI metrics will appear here once data is available.')).toBeInTheDocument();
  });

  it('renders loading state when isLoading is true', () => {
    render(
      <AllTheProviders>
        <KPIGrid isLoading={true} />
      </AllTheProviders>
    );

    expect(screen.queryByText('No KPI Data Available')).not.toBeInTheDocument();
  });

  it('renders error state when error is provided', () => {
    render(
      <AllTheProviders>
        <KPIGrid error={new Error('Test error')} />
      </AllTheProviders>
    );

    expect(screen.getByText('Error Loading KPIs')).toBeInTheDocument();
  });
});
