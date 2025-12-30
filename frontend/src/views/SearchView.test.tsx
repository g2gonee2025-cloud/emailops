
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import SearchView from './SearchView';
import { AllTheProviders } from '../tests/testUtils'; // Wrapper for context providers

// Mock the hooks and modules
vi.mock('../hooks/useSearch', () => ({
  useSearch: vi.fn(),
}));

vi.mock('../hooks/useDebounce', () => ({
  // Return the value immediately for testing
  useDebounce: (value: unknown) => value,
}));

vi.mock('../components/ui/Toast', async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    useToast: () => ({
      toast: vi.fn(),
    }),
  };
});

// Re-import after mocks are set
const { useSearch } = await import('../hooks/useSearch');

const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

const mockData = {
  results: [
    {
      chunk_id: '1',
      conversation_id: 'conv1',
      thread_id: 'thread1',
      content: 'This is the first search result.',
      score: 0.95,
    },
    {
      chunk_id: '2',
      conversation_id: 'conv2',
      content: 'This is the second search result.',
      score: 0.85,
    },
  ],
  total_count: 2,
  query_time_ms: 42,
};

describe('SearchView', () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  const renderComponent = () => {
    return render(
      <AllTheProviders>
        <SearchView />
      </AllTheProviders>,
    );
  };

  it('renders the initial state correctly', () => {
    (useSearch as vi.Mock).mockReturnValue({
      data: null,
      isLoading: false,
      isError: false,
    });
    renderComponent();

    expect(screen.getByText('Start Searching')).toBeInTheDocument();
    expect(
      screen.getByText(/Find relevant emails, conversations, and documents/),
    ).toBeInTheDocument();
  });

  it('displays a loading state while fetching results', () => {
    (useSearch as vi.Mock).mockReturnValue({
      data: null,
      isLoading: true,
      isError: false,
    });
    renderComponent();

    fireEvent.change(screen.getByLabelText('Search query'), {
      target: { value: 'test' },
    });

    expect(screen.getByRole('status', { name: /loading search results/i })).toBeInTheDocument();
  });

  it('displays search results when the query is successful', async () => {
    (useSearch as vi.Mock).mockReturnValue({
      data: mockData,
      isLoading: false,
      isError: false,
    });
    renderComponent();

    fireEvent.change(screen.getByLabelText('Search query'), {
      target: { value: 'test query' },
    });

    await waitFor(() => {
      expect(screen.getByText('This is the first search result.')).toBeInTheDocument();
      expect(screen.getByText('This is the second search result.')).toBeInTheDocument();
      expect(screen.getByText(/2 results/)).toBeInTheDocument();
      expect(screen.getByText(/42ms/)).toBeInTheDocument();
      expect(screen.getByText(/95% match/)).toBeInTheDocument();
    });
  });

  it('displays a "no results" message when the search yields no results', async () => {
    (useSearch as vi.Mock).mockReturnValue({
      data: { results: [], total_count: 0, query_time_ms: 10 },
      isLoading: false,
      isError: false,
    });
    renderComponent();

    fireEvent.change(screen.getByLabelText('Search query'), {
      target: { value: 'empty' },
    });

    await waitFor(() => {
      expect(screen.getByText('No Results Found')).toBeInTheDocument();
    });
  });

  it('displays an error message when the search fails', async () => {
    (useSearch as vi.Mock).mockReturnValue({
      data: null,
      isLoading: false,
      isError: true,
      error: new Error('Failed to fetch'),
    });
    renderComponent();

    fireEvent.change(screen.getByLabelText('Search query'), {
      target: { value: 'error' },
    });

    await waitFor(() => {
      expect(screen.getByText('Error')).toBeInTheDocument();
      expect(
        screen.getByText('There was a problem with your search. Please try again later.'),
      ).toBeInTheDocument();
    });
  });

  it('navigates to the correct thread when a result is clicked', async () => {
    (useSearch as vi.Mock).mockReturnValue({
      data: mockData,
      isLoading: false,
      isError: false,
    });
    renderComponent();

    fireEvent.change(screen.getByLabelText('Search query'), {
      target: { value: 'navigate' },
    });

    const resultText = await screen.findByText('This is the first search result.');
    fireEvent.click(resultText);

    expect(mockNavigate).toHaveBeenCalledWith('/thread/thread1');
  });

  it('clears the input when the clear button is clicked', async () => {
    (useSearch as vi.Mock).mockReturnValue({
      data: null,
      isLoading: false,
      isError: false,
    });
    renderComponent();

    const input = screen.getByLabelText('Search query');
    fireEvent.change(input, { target: { value: 'some query' } });

    await screen.findByDisplayValue('some query');

    const clearButton = screen.getByLabelText('Clear search');
    fireEvent.click(clearButton);

    await screen.findByDisplayValue('');
    expect(input).toHaveValue('');
  });
});
