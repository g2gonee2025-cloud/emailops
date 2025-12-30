
import { render, screen, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import userEvent from '@testing-library/user-event';
import DraftView from '../views/DraftView';
import { AllTheProviders } from './testUtils';
import { useDraftEmail } from '../hooks/useDraft';
import { type DraftEmailResponse, type ApiError } from '../lib/api';
import React from 'react';

// Mock the useDraft hook
vi.mock('../hooks/useDraft', () => ({
  useDraftEmail: vi.fn(),
}));

// Mock the useToast hook and ToastProvider
const mockAddToast = vi.fn();
vi.mock('../contexts/toastContext', () => ({
  useToast: () => ({
    addToast: mockAddToast,
  }),
  ToastProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

const mockMutate = vi.fn();
const mockReset = vi.fn();

describe('DraftView', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Provide a default mock implementation for every test
    vi.mocked(useDraftEmail).mockReturnValue({
      mutate: mockMutate,
      data: null,
      error: null,
      isPending: false,
      reset: mockReset,
    } as any);
  });

  const renderComponent = () =>
    render(
      <AllTheProviders>
        <DraftView />
      </AllTheProviders>
    );

  it('renders the initial empty state correctly', () => {
    renderComponent();

    expect(screen.getByText('Compose with AI')).toBeInTheDocument();
    expect(
      screen.getByText('Describe what you want to say, and Cortex will draft a polished email for you.')
    ).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /generate draft/i })).toBeInTheDocument();
  });

  it('shows the loading skeleton when drafting is in progress', () => {
    vi.mocked(useDraftEmail).mockReturnValue({
      mutate: mockMutate,
      data: null,
      error: null,
      isPending: true,
      reset: mockReset,
    } as any);

    renderComponent();

    // The Skeleton component from shadcn/ui doesn't have an implicit role.
    // We check for the presence of multiple elements with the animation class.
    const skeletons = screen.getAllByText((content, element) => {
      return element?.classList.contains('animate-pulse') ?? false;
    });
    expect(skeletons.length).toBeGreaterThan(5);
  });

  it('displays the generated draft upon success', async () => {
    const mockDraft: DraftEmailResponse = {
      draft: {
        subject: 'Test Subject',
        body: 'This is the test body of the email.',
        to: ['test@example.com'],
        cc: [],
      },
    };
    vi.mocked(useDraftEmail).mockReturnValue({
      mutate: mockMutate,
      data: mockDraft,
      error: null,
      isPending: false,
      reset: mockReset,
    } as any);

    renderComponent();

    await waitFor(() => {
      expect(screen.getByText('Generated Draft')).toBeInTheDocument();
    });
    expect(screen.getByText('Test Subject')).toBeInTheDocument();
    expect(screen.getByText('This is the test body of the email.')).toBeInTheDocument();
    expect(screen.getByText('test@example.com')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /copy/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /new draft/i })).toBeInTheDocument();
  });

  it('displays an error message when drafting fails and shows an alert', async () => {
    const mockError: ApiError = {
      message: 'Failed to generate draft.',
    };
    vi.mocked(useDraftEmail).mockReturnValue({
      mutate: mockMutate,
      data: null,
      error: mockError,
      isPending: false,
      reset: mockReset,
    } as any);

    renderComponent();

    // Check if the toast is called
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'error',
        message: 'Error Generating Draft',
        details: 'Failed to generate draft.',
      });
    });

    // Check if the inline alert is displayed
    expect(await screen.findByRole('alert')).toBeInTheDocument();
    expect(screen.getByText(mockError.message)).toBeInTheDocument();
  });

  it('calls the mutate function with form data on submit', async () => {
    const user = userEvent.setup();
    renderComponent();

    const instructionInput = screen.getByPlaceholderText(/e.g., Reply to John/i);
    const threadIdInput = screen.getByPlaceholderText(/Leave empty to draft without context/i);
    const submitButton = screen.getByRole('button', { name: /generate draft/i });

    await user.type(instructionInput, 'Test instruction');
    await user.type(threadIdInput, 'thread-123');
    await user.click(submitButton);

    expect(mockMutate).toHaveBeenCalledWith({
      instruction: 'Test instruction',
      threadId: 'thread-123',
      tone: 'professional', // Default tone
    });
  });

  it('resets the form and mutation when "New Draft" is clicked', async () => {
    const user = userEvent.setup();
    const mockDraft: DraftEmailResponse = {
      draft: { subject: 'Old Subject', body: 'Old body' },
    };

    let data: DraftEmailResponse | null = mockDraft;

    const mockResetFn = () => {
      data = null; // This closure will modify the `data` variable
      mockReset();
    };

    vi.mocked(useDraftEmail).mockImplementation(() => ({
      mutate: mockMutate,
      data: data,
      error: null,
      isPending: false,
      reset: mockResetFn,
    }) as any);

    const { rerender } = renderComponent();

    // Wait for the draft to be displayed
    await waitFor(() => {
      expect(screen.getByText('Generated Draft')).toBeInTheDocument();
    });

    const newDraftButton = screen.getByRole('button', { name: /new draft/i });
    await user.click(newDraftButton);

    // After the click, the component's internal state (from react-hook-form) changes,
    // which triggers a re-render. During the re-render, our mock is called again,
    // this time returning `data: null` because the `reset` function was called.
    rerender(<AllTheProviders><DraftView /></AllTheProviders>);

    // Assert that the reset function was called
    expect(mockReset).toHaveBeenCalled();

    // Assert that the view has returned to the initial empty state.
    await waitFor(() => {
      expect(screen.getByText('Compose with AI')).toBeInTheDocument();
    });
    expect(screen.getByPlaceholderText(/e.g., Reply to John/i)).toBeInTheDocument();
  });
});
