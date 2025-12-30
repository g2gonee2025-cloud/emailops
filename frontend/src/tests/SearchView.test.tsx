
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import SearchView from '../views/SearchView';
import { AllTheProviders } from './testUtils';
import { useSearch } from '../hooks/useSearch';
import { useDebounce } from '../hooks/useDebounce';
import { type SearchResponse } from '../schemas/search';

// Mock hooks
vi.mock('../hooks/useSearch');
vi.mock('../hooks/useDebounce');

// Mock react-router-dom
const mockedNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...(actual as object),
    useNavigate: () => mockedNavigate,
  };
});

// Mock the useToast hook
const mockedAddToast = vi.fn();
vi.mock('../contexts/toastContext', async () => {
  const actual = await vi.importActual('../contexts/toastContext');
  return {
    ...(actual as object),
    useToast: () => ({
      addToast: mockedAddToast,
    }),
  };
});

const mockUseSearch = useSearch as jest.Mock;
const mockUseDebounce = useDebounce as jest.Mock;

const renderComponent = () => {
  return render(
    <AllTheProviders>
      <SearchView />
    </AllTheProviders>
  );
};

describe('SearchView', () => {
  beforeEach(() => {
    vi.resetAllMocks();

    // Mock useDebounce to return the value immediately
    mockUseDebounce.mockImplementation((value) => value);

    // Default mock for useSearch
    mockUseSearch.mockReturnValue({
      data: null,
      isLoading: false,
      isError: false,
      error: null,
    });
  });

  it('renders the initial state with no query', () => {
    renderComponent();
    expect(screen.getByText('Start Searching')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Search emails, conversations, claims...')).toBeInTheDocument();
  });

  it('shows a loading skeleton when searching', async () => {
    mockUseSearch.mockReturnValue({
      data: null,
      isLoading: true,
      isError: false,
      error: null,
    });
    renderComponent();

    const searchInput = screen.getByPlaceholderText('Search emails, conversations, claims...');
    fireEvent.change(searchInput, { target: { value: 'test query' } });

    await waitFor(() => {
      expect(screen.getByRole('status', { name: /loading search results/i })).toBeInTheDocument();
    });
  });

  it('displays an error message when the search fails', async () => {
    const testError = new Error('Network Error');
    mockUseSearch.mockReturnValue({
      data: null,
      isLoading: false,
      isError: true,
      error: testError,
    });
    renderComponent();

    const searchInput = screen.getByPlaceholderText('Search emails, conversations, claims...');
    fireEvent.change(searchInput, { target: { value: 'failing query' } });

    await waitFor(() => {
      expect(screen.getByText('Error')).toBeInTheDocument();
      expect(screen.getByText('There was a problem with your search. Please try again later.')).toBeInTheDocument();
      expect(mockedAddToast).toHaveBeenCalledWith({
        type: 'error',
        message: 'Search Failed',
        details: 'Network Error',
      });
    });
  });

  it('shows a "No Results Found" message for an empty result set', async () => {
    const mockEmptyData: SearchResponse = {
        results: [],
        total_count: 0,
        query_time_ms: 15,
    };
    mockUseSearch.mockReturnValue({
      data: mockEmptyData,
      isLoading: false,
      isError: false,
      error: null,
    });
    renderComponent();

    const searchInput = screen.getByPlaceholderText('Search emails, conversations, claims...');
    fireEvent.change(searchInput, { target: { value: 'empty query' } });

    await waitFor(() => {
        expect(screen.getByText('No Results Found')).toBeInTheDocument();
    });
  });

  it('displays search results successfully', async () => {
    const mockSuccessData: SearchResponse = {
        total_count: 1,
        query_time_ms: 42,
        results: [
            {
              chunk_id: 'chunk1',
              content: 'This is the content of the first search result.',
              score: 0.95,
              thread_id: 'thread123',
              conversation_id: 'conv456',
            },
          ],
    };
    mockUseSearch.mockReturnValue({
      data: mockSuccessData,
      isLoading: false,
      isError: false,
      error: null,
    });
    renderComponent();

    const searchInput = screen.getByPlaceholderText('Search emails, conversations, claims...');
    fireEvent.change(searchInput, { target: { value: 'success query' } });

    await waitFor(() => {
      expect(screen.getByText('This is the content of the first search result.')).toBeInTheDocument();
      expect(screen.getByText('95% match')).toBeInTheDocument();
      expect(screen.getByText('1 results')).toBeInTheDocument();
      expect(screen.getByText('• 42ms')).toBeInTheDocument();
    });
  });

  it('navigates to the correct thread when a result is clicked', async () => {
    const mockSuccessData: SearchResponse = {
        total_count: 1,
        query_time_ms: 42,
        results: [
            {
              chunk_id: 'chunk1',
              content: 'Clickable result.',
              score: 0.88,
              thread_id: 'thread123',
              conversation_id: 'conv456',
            },
          ],
    };
    mockUseSearch.mockReturnValue({
      data: mockSuccessData,
      isLoading: false,
      isError: false,
      error: null,
    });
    renderComponent();

    const searchInput = screen.getByPlaceholderText('Search emails, conversations, claims...');
    fireEvent.change(searchInput, { target: { value: 'nav query' } });

    await waitFor(async () => {
        const resultElement = await screen.findByText('Clickable result.');
        // The onClick is on the parent GlassCard, event will bubble up
        fireEvent.click(resultElement);
        expect(mockedNavigate).toHaveBeenCalledWith('/thread/thread123');
      });
  });

  it('clears the search input when the clear button is clicked', async () => {
    renderComponent();
    const searchInput = screen.getByPlaceholderText('Search emails, conversations, claims...') as HTMLInputElement;

    fireEvent.change(searchInput, { target: { value: 'some query' } });
    expect(searchInput.value).toBe('some query');

    const clearButton = await screen.findByRole('button', { name: /clear search/i });
    fireEvent.click(clearButton);
    expect(searchInput.value).toBe('');
  });
});
