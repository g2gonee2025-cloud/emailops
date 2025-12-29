import { render, screen } from '@testing-library/react';
import { Label } from './Label';
import { describe, it, expect } from 'vitest';

describe('Label', () => {
  it('renders the label with the correct text', () => {
    render(<Label>Test Label</Label>);
    const labelElement = screen.getByText('Test Label');
    expect(labelElement).toBeInTheDocument();
  });
});
