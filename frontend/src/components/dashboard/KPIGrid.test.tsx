import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import KPIGrid from './KPIGrid';
import { AllTheProviders } from '../../tests/testUtils';

describe('KPIGrid', () => {
  it('renders the KPI cards with mock data', () => {
    render(
      <AllTheProviders>
        <KPIGrid />
      </AllTheProviders>
    );

    // Check for titles
    expect(screen.getByText('Total Emails Processed')).toBeInTheDocument();
    expect(screen.getByText('Average Response Time')).toBeInTheDocument();
    expect(screen.getByText('Open Rate')).toBeInTheDocument();
    expect(screen.getByText('Click-Through Rate')).toBeInTheDocument();

    // Check for values
    expect(screen.getByText('1,234,567')).toBeInTheDocument();
    expect(screen.getByText('2.3 hours')).toBeInTheDocument();
    expect(screen.getByText('25.8%')).toBeInTheDocument();
    expect(screen.getByText('4.2%')).toBeInTheDocument();

    // Check for changes
    expect(screen.getByText('+5.2%')).toBeInTheDocument();
    expect(screen.getByText('-1.5%')).toBeInTheDocument();
    expect(screen.getByText('+0.8%')).toBeInTheDocument();
    expect(screen.getByText('+0.2%')).toBeInTheDocument();
  });
});
