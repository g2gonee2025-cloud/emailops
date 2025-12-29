import { render, screen } from '@testing-library/react';
import { useQueryClient } from '@tanstack/react-query';
import { QueryProvider } from './queryClient';

// A simple component to test if the query client is available
const TestComponent = () => {
  const queryClient = useQueryClient();
  return <div>{queryClient ? 'Client available' : 'Client not available'}</div>;
};

describe('QueryProvider', () => {
  it('should provide a query client to its children', () => {
    render(
      <QueryProvider>
        <TestComponent />
      </QueryProvider>
    );

    expect(screen.getByText('Client available')).toBeInTheDocument();
  });
});
