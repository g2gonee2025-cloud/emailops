import { render, screen, fireEvent } from '@testing-library/react';
import ErrorBoundary from './ErrorBoundary';

// A component that throws an error
const ProblemChild = () => {
  throw new Error('Test Error');
};

// A component that renders without error
const GoodChild = () => <div>No Error</div>;

describe('ErrorBoundary', () => {
  // Suppress console.error output for this test suite
  beforeAll(() => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterAll(() => {
    vi.restoreAllMocks();
  });

  it('should render children when there is no error', () => {
    render(
      <ErrorBoundary>
        <GoodChild />
      </ErrorBoundary>
    );
    expect(screen.getByText('No Error')).toBeInTheDocument();
  });

  it('should render the fallback UI when an error is thrown', () => {
    render(
      <ErrorBoundary>
        <ProblemChild />
      </ErrorBoundary>
    );
    expect(screen.getByText('Something went wrong.')).toBeInTheDocument();
    expect(screen.getByText("We're sorry for the inconvenience. Please try again.")).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Try again' })).toBeInTheDocument();
  });

  it('should reset the error state and re-render children when "Try again" is clicked', () => {
    const { rerender } = render(
      <ErrorBoundary>
        <ProblemChild />
      </ErrorBoundary>
    );

    // Verify fallback UI is shown
    expect(screen.getByText('Something went wrong.')).toBeInTheDocument();

    // Re-render with a non-erroring child
    rerender(
        <ErrorBoundary>
          <GoodChild />
        </ErrorBoundary>
      );

    // Click the "Try again" button
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }));

    // Verify the children are rendered again
    expect(screen.getByText('No Error')).toBeInTheDocument();
  });
});
