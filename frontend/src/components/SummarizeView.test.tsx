import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import SummarizeView from './SummarizeView';
import { AllTheProviders } from '../tests/testUtils';
import { useSummarize } from '../hooks/useSummarize';
import { ThreadSummary } from '../schemas/summarize';

// Mock hooks
vi.mock('../hooks/useSummarize');

// Partially mock the Toast module to override useToast but keep other exports
vi.mock('./ui/Toast', async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    useToast: () => ({
      toast: vi.fn(),
    }),
  };
});

const mockUseSummarize = useSummarize as vi.Mock;

describe('SummarizeView', () => {
  const mockSummarize = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders the summarize button and calls summarize on click', () => {
    mockUseSummarize.mockReturnValue({
      summarize: mockSummarize,
      isLoading: false,
      error: null,
      data: null,
    });

    render(
      <AllTheProviders>
        <SummarizeView threadId="test-thread-id" />
      </AllTheProviders>
    );

    const summarizeButton = screen.getByText('Summarize');
    expect(summarizeButton).toBeInTheDocument();
    fireEvent.click(summarizeButton);
    expect(mockSummarize).toHaveBeenCalledWith({ threadId: 'test-thread-id' });
  });

  it('displays a loading skeleton when summarizing', () => {
    mockUseSummarize.mockReturnValue({
      summarize: mockSummarize,
      isLoading: true,
      error: null,
      data: null,
    });

    render(
      <AllTheProviders>
        <SummarizeView threadId="test-thread-id" />
      </AllTheProviders>
    );

    expect(screen.getByText('Generating Summary...')).toBeInTheDocument();
  });

  it('displays the summary on success', async () => {
    const summaryData: { summary: ThreadSummary } = {
      summary: {
        summary: 'This is a test summary.',
        key_points: ['Point 1', 'Point 2'],
      },
    };

    mockUseSummarize.mockReturnValue({
      summarize: mockSummarize,
      isLoading: false,
      error: null,
      data: summaryData,
    });

    render(
      <AllTheProviders>
        <SummarizeView threadId="test-thread-id" />
      </AllTheProviders>
    );

    await waitFor(() => {
      expect(screen.getByText('AI Summary')).toBeInTheDocument();
      expect(screen.getByText('This is a test summary.')).toBeInTheDocument();
      expect(screen.getByText('Point 1')).toBeInTheDocument();
      expect(screen.getByText('Point 2')).toBeInTheDocument();
    });
  });

  it('displays an error message on failure', async () => {
    mockUseSummarize.mockReturnValue({
      summarize: mockSummarize,
      isLoading: false,
      error: new Error('Summarization failed'),
      data: null,
    });

    render(
      <AllTheProviders>
        <SummarizeView threadId="test-thread-id" />
      </AllTheProviders>
    );

    await waitFor(() => {
      expect(screen.getByText('Summarization Error')).toBeInTheDocument();
    });
  });
});
