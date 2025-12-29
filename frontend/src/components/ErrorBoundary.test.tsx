import { render, screen } from '@testing-library/react';
import ErrorBoundary from './ErrorBoundary';
import { expect, test, vi } from 'vitest';

const ThrowError = () => {
    throw new Error('Test error');
};

test('ErrorBoundary catches error and displays fallback UI', () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});

    render(
        <ErrorBoundary>
            <ThrowError />
        </ErrorBoundary>
    );

    expect(screen.getByText(/System Malfunction/i)).toBeInTheDocument();
    expect(screen.getByText(/Reboot System/i)).toBeInTheDocument();
});
