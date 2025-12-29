/** @vitest-environment jsdom */
import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import GlassCard from './GlassCard';

describe('GlassCard', () => {
  afterEach(() => {
    cleanup();
  });

  it('should render children correctly', () => {
    render(
      <GlassCard>
        <div>Test Child</div>
      </GlassCard>
    );
    expect(screen.getByText('Test Child')).toBeInTheDocument();
  });

  it('should apply medium intensity styles by default', () => {
    const { container } = render(<GlassCard>Default</GlassCard>);
    // The first child of the container is the div we are interested in.
    expect(container.firstChild).toHaveClass('bg-white/10 backdrop-blur-md');
  });

  it('should apply low intensity styles when specified', () => {
    const { container } = render(<GlassCard intensity="low">Low</GlassCard>);
    expect(container.firstChild).toHaveClass('bg-white/5 backdrop-blur-sm');
  });

  it('should apply high intensity styles when specified', () => {
    const { container } = render(<GlassCard intensity="high">High</GlassCard>);
    expect(container.firstChild).toHaveClass('bg-white/20 backdrop-blur-lg');
  });

  it('should not apply hover effect styles by default', () => {
    const { container } = render(<GlassCard>No Hover</GlassCard>);
    expect(container.firstChild).not.toHaveClass('hover:bg-white/15');
  });

  it('should apply hover effect styles when enabled', () => {
    const { container } = render(<GlassCard hoverEffect>Hover</GlassCard>);
    expect(container.firstChild).toHaveClass('hover:bg-white/15 hover:scale-[1.02] hover:shadow-2xl hover:border-white/20');
  });

  it('should pass through additional HTML attributes', () => {
    render(<GlassCard data-testid="test-card" id="my-card">Attributes</GlassCard>);
    const card = screen.getByTestId('test-card');
    expect(card).toBeInTheDocument();
    expect(card).toHaveAttribute('id', 'my-card');
  });
});
