import { render, screen } from '@testing-library/react';
import { useQueryClient } from '@tanstack/react-query';
import { QueryProvider, queryClient } from './queryClient.tsx';

const TestComponent = () => {
  const client = useQueryClient();
  return <div>{client === queryClient ? 'Client Provided' : 'Error'}</div>;
};

describe('QueryProvider', () => {
  it('provides the query client to its children', () => {
    render(
      <QueryProvider>
        <TestComponent />
      </QueryProvider>
    );

    expect(screen.getByText('Client Provided')).toBeInTheDocument();
  });
});
