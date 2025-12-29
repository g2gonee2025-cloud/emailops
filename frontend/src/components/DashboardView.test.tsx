import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { DashboardView } from './DashboardView';

describe('DashboardView', () => {
  it('renders the main heading', () => {
    render(<DashboardView />);
    expect(screen.getByRole('heading', { name: /mission control/i })).toBeInTheDocument();
  });
});
