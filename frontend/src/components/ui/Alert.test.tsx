import { Terminal } from 'lucide-react';
import { describe, expect, it } from 'vitest';

import { render, screen } from '@/tests/testUtils';
import { Alert, AlertDescription, AlertTitle } from './Alert';

describe('Alert', () => {
  it('should render correctly with default variant', () => {
    render(
      <Alert>
        <Terminal className="h-4 w-4" />
        <AlertTitle>Heads up!</AlertTitle>
        <AlertDescription>
          You can add components to your app using the cli.
        </AlertDescription>
      </Alert>,
    );

    expect(screen.getByText('Heads up!')).toBeInTheDocument();
    expect(
      screen.getByText('You can add components to your app using the cli.'),
    ).toBeInTheDocument();
  });

  it('should render correctly with destructive variant', () => {
    render(
      <Alert variant="destructive">
        <Terminal className="h-4 w-4" />
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>
          Your session has expired. Please log in again.
        </AlertDescription>
      </Alert>,
    );

    expect(screen.getByText('Error')).toBeInTheDocument();
    expect(
      screen.getByText('Your session has expired. Please log in again.'),
    ).toBeInTheDocument();

    const alertElement = screen.getByRole('alert');
    expect(alertElement).toHaveClass(
      'border-red-500/50 text-red-500 dark:border-red-500 [&>svg]:text-red-500',
    );
  });
});
