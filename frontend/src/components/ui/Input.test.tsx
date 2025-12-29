/**
 * @vitest-environment jsdom
 */
import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Input } from './Input';

describe('Input', () => {
  afterEach(() => {
    cleanup();
  });

  it('should render an input element', () => {
    render(<Input data-testid="test-input" />);
    const inputElement = screen.getByTestId('test-input');
    expect(inputElement).toBeInTheDocument();
  });

  it('should handle user input correctly', async () => {
    const user = userEvent.setup();
    render(<Input data-testid="test-input" />);
    const inputElement = screen.getByTestId('test-input') as HTMLInputElement;

    await user.type(inputElement, 'Hello, world!');
    expect(inputElement.value).toBe('Hello, world!');
  });

  it('should be disabled when the disabled prop is true', () => {
    render(<Input disabled data-testid="test-input" />);
    const inputElement = screen.getByTestId('test-input');
    expect(inputElement).toBeDisabled();
  });

  it('should have the correct type attribute', () => {
    render(<Input type="password" data-testid="test-input" />);
    const inputElement = screen.getByTestId('test-input');
    expect(inputElement).toHaveAttribute('type', 'password');
  });

  it('should display a placeholder', () => {
    const placeholderText = 'Enter your name';
    render(<Input placeholder={placeholderText} />);
    const inputElement = screen.getByPlaceholderText(placeholderText);
    expect(inputElement).toBeInTheDocument();
  });

  it('should apply additional classNames', () => {
    const customClass = 'my-custom-class';
    render(<Input className={customClass} data-testid="test-input" />);
    const inputElement = screen.getByTestId('test-input');
    expect(inputElement).toHaveClass(customClass);
  });
});
