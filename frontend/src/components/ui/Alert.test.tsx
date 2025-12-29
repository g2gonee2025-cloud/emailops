/** @vitest-environment jsdom */
import { render, screen } from '@testing-library/react';
import { describe, it, expect, afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';
import * as matchers from '@testing-library/jest-dom/matchers';
import { Alert, AlertTitle, AlertDescription } from './Alert';

expect.extend(matchers);

describe('Alert component', () => {
  afterEach(() => {
    cleanup();
  });

  it('renders correctly with title and description', () => {
    render(
      <Alert>
        <AlertTitle>Test Title</AlertTitle>
        <AlertDescription>Test Description</AlertDescription>
      </Alert>,
    );

    expect(screen.getByText('Test Title')).toBeInTheDocument();
    expect(screen.getByText('Test Description')).toBeInTheDocument();
  });

  it('has the correct role attribute for accessibility', () => {
    render(<Alert />);
    expect(screen.getByRole('alert')).toBeInTheDocument();
  });

  it('applies the default variant styles correctly', () => {
    render(<Alert data-testid="alert" />);
    const alertElement = screen.getByTestId('alert');
    expect(alertElement).toHaveClass('bg-background');
    expect(alertElement).toHaveClass('text-foreground');
  });

  it('applies the destructive variant styles correctly', () => {
    render(<Alert variant="destructive" data-testid="alert" />);
    const alertElement = screen.getByTestId('alert');
    expect(alertElement).toHaveClass('border-destructive/50');
    expect(alertElement).toHaveClass('text-destructive');
  });
});
