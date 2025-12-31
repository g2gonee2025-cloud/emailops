/** @vitest-environment jsdom */
import { render, screen, cleanup } from '@testing-library/react';
import { Progress } from './Progress';
import { describe, it, expect, afterEach } from 'vitest';

describe('Progress', () => {
  afterEach(() => {
    cleanup();
  });

  it('renders with default props', () => {
    render(<Progress />);
    const progress = screen.getByRole('progressbar');
    expect(progress).toBeInTheDocument();
    expect(progress.getAttribute('aria-valuenow')).toBe('0');
  });

  it('renders with a specific value and max', () => {
    render(<Progress value={50} max={100} />);
    const progress = screen.getByRole('progressbar');
    expect(progress.getAttribute('aria-valuenow')).toBe('50');
    expect(progress.getAttribute('aria-valuemax')).toBe('100');
  });

  it('renders with a label and value', () => {
    render(<Progress value={25} label="Loading" showValue />);
    expect(screen.getByText('Loading')).toBeInTheDocument();
    expect(screen.getByText('25%')).toBeInTheDocument();
  });

  it('applies the correct size class', () => {
    render(<Progress value={50} size="large" />);
    const progress = screen.getByRole('progressbar');
    expect(progress).toHaveClass('h-4');
  });

  it('applies the correct variant class', () => {
    render(<Progress value={50} variant="success" />);
    const progressIndicator = screen.getByRole('progressbar').firstChild;
    expect(progressIndicator).toHaveClass('bg-green-500');
  });
});
