import { render, screen, fireEvent } from '@testing-library/react';
import { MemoryRouter, useSearchParams } from 'react-router-dom';
import { describe, it, expect, vi, type Mock } from 'vitest';

import { FilterBar } from './FilterBar';

const routerFuture = {
  v7_startTransition: true,
  v7_relativeSplatPath: true,
};

vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom') as object;
  return {
    ...actual,
    useSearchParams: vi.fn(),
  };
});

const mockUseSearchParams = useSearchParams as Mock;

describe('FilterBar', () => {
  it('renders and updates search params on filter change', async () => {
    const setSearchParams = vi.fn();
    mockUseSearchParams.mockReturnValue([new URLSearchParams(), setSearchParams]);

    render(
      <MemoryRouter future={routerFuture}>
        <FilterBar />
      </MemoryRouter>
    );

    expect(screen.getByText('File Type')).toBeInTheDocument();

    const selectTrigger = screen.getByRole('combobox');
    fireEvent.click(selectTrigger);

    const option = await screen.findByText('Email');
    fireEvent.click(option);

    expect(setSearchParams).toHaveBeenCalled();
  });
});
