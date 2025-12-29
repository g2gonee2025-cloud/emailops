import { screen, fireEvent } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { Checkbox } from './Checkbox';
import { renderWithProviders } from '../../tests/testUtils';


describe('Checkbox', () => {
  it('should render in an unchecked state by default', () => {
    renderWithProviders(<Checkbox id="test-checkbox" />);
    const checkbox = screen.getByRole('checkbox');
    expect(checkbox).not.toBeChecked();
  });

  it('should toggle state to checked when clicked', () => {
    renderWithProviders(<Checkbox id="test-checkbox" />);
    const checkbox = screen.getByRole('checkbox');
    fireEvent.click(checkbox);
    expect(checkbox).toBeChecked();
  });

  it('should be disabled when the disabled prop is true', () => {
    renderWithProviders(<Checkbox id="test-checkbox" disabled />);
    const checkbox = screen.getByRole('checkbox');
    expect(checkbox).toBeDisabled();
  });
});
