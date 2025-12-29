import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { Input } from './Input';

describe('Input', () => {
  it('renders without crashing', () => {
    render(<Input data-testid="test-input" />);
    const inputElement = screen.getByTestId('test-input');
    expect(inputElement).toBeInTheDocument();
    expect(inputElement.tagName).toBe('INPUT');
  });

  it('applies default medium size styles', () => {
    render(<Input data-testid="test-input" />);
    const inputElement = screen.getByTestId('test-input');
    expect(inputElement).toHaveClass('h-10', 'px-3', 'py-2', 'text-sm');
  });

  it('applies small size styles', () => {
    render(<Input size="sm" data-testid="test-input" />);
    const inputElement = screen.getByTestId('test-input');
    expect(inputElement).toHaveClass('h-9', 'px-3', 'py-1', 'text-xs');
  });

  it('applies large size styles', () => {
    render(<Input size="lg" data-testid="test-input" />);
    const inputElement = screen.getByTestId('test-input');
    expect(inputElement).toHaveClass('h-11', 'px-4', 'py-2', 'text-base');
  });

  it('forwards ref to the input element', () => {
    const ref = React.createRef<HTMLInputElement>();
    render(<Input ref={ref} />);
    expect(ref.current).not.toBeNull();
    expect(ref.current?.tagName).toBe('INPUT');
  });

  it('applies custom className', () => {
    const customClass = 'my-custom-class';
    render(<Input className={customClass} data-testid="test-input" />);
    const inputElement = screen.getByTestId('test-input');
    expect(inputElement).toHaveClass(customClass);
  });

  it('passes other props to the input element', () => {
    render(<Input placeholder="Enter text" disabled data-testid="test-input" />);
    const inputElement = screen.getByTestId('test-input');
    expect(inputElement).toHaveAttribute('placeholder', 'Enter text');
    expect(inputElement).toBeDisabled();
  });
});
