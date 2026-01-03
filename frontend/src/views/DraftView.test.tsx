
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi, type Mock } from 'vitest';
import DraftView from './DraftView';
import { AllTheProviders } from '../tests/testUtils';
import { useDraftEmail } from '../hooks/useDraft';

// Mock the entire hooks/useDraft module
vi.mock('../hooks/useDraft');

// Partially mock the contexts/toastContext module to spy on useToast
const mockAddToast = vi.fn();
vi.mock('../contexts/toastContext', async (importOriginal) => {
    const actual = await importOriginal() as object;
    return {
        ...actual,
        useToast: () => ({
            addToast: mockAddToast,
        }),
    };
});

// Cast the imported hook to a Mock type for TypeScript support
const useDraftEmailMock = useDraftEmail as Mock;

describe('DraftView', () => {
    const mockMutate = vi.fn();
    const mockReset = vi.fn();

    beforeEach(() => {
        // Clear all mocks before each test
        vi.clearAllMocks();
        // Set a default return value for the mocked hook
        useDraftEmailMock.mockReturnValue({
            mutate: mockMutate,
            data: null,
            error: null,
            isPending: false,
            reset: mockReset,
        });
    });

    const renderComponent = () =>
        render(
            <AllTheProviders>
                <DraftView />
            </AllTheProviders>
        );

    it('renders the initial form correctly', () => {
        renderComponent();
        expect(screen.getByText('Draft Email')).toBeInTheDocument();
        expect(screen.getByLabelText(/What should the email say?/i)).toBeInTheDocument();
        // ConversationSelector is now a dropdown, check for the label instead
        expect(screen.getByText(/Conversation \(optional\)/i)).toBeInTheDocument();
        expect(screen.getByText('Generate Draft')).toBeInTheDocument();
    });

    it('shows a validation error if instruction is empty on submit', async () => {
        renderComponent();
        fireEvent.click(screen.getByText('Generate Draft'));

        await waitFor(() => {
            expect(screen.getByText('Instruction cannot be empty.')).toBeInTheDocument();
        });
        expect(mockMutate).not.toHaveBeenCalled();
    });

    it('calls the draftEmail mutation on form submission', async () => {
        renderComponent();
        const instructionInput = screen.getByLabelText(/What should the email say?/i);
        const submitButton = screen.getByText('Generate Draft');

        const testInstruction = 'Test instruction';

        fireEvent.change(instructionInput, { target: { value: testInstruction } });
        // Note: ConversationSelector is now a dropdown, not a text input
        // Testing without selecting a conversation (threadId will be empty)
        fireEvent.click(submitButton);

        await waitFor(() => {
            expect(mockMutate).toHaveBeenCalledWith({
                instruction: testInstruction,
                threadId: '',
                tone: 'professional',
            });
        });
    });

    it('displays the generated draft on success', async () => {
        const mockDraft = {
            subject: 'Test Subject',
            body: 'Test Body',
            to: ['test@example.com'],
            cc: [],
        };
        // Override the mock return value for this specific test
        useDraftEmailMock.mockReturnValue({
            mutate: mockMutate,
            data: { draft: mockDraft },
            error: null,
            isPending: false,
            reset: mockReset,
        });

        renderComponent();

        await waitFor(() => {
            expect(screen.getByText('Generated Draft')).toBeInTheDocument();
            expect(screen.getByText(mockDraft.subject)).toBeInTheDocument();
            expect(screen.getByText(mockDraft.body)).toBeInTheDocument();
            expect(screen.getByText(mockDraft.to[0])).toBeInTheDocument();
        });
    });

    it('displays an error message and a toast on failure', async () => {
        const errorMessage = 'Failed to generate draft';
        const error = new Error(errorMessage);
        // Override the mock return value for this specific test
        useDraftEmailMock.mockReturnValue({
            mutate: mockMutate,
            data: null,
            error: error,
            isPending: false,
            reset: mockReset,
        });

        renderComponent();

        await waitFor(() => {
            expect(screen.getByText(errorMessage)).toBeInTheDocument();
            expect(mockAddToast).toHaveBeenCalledWith({
                type: 'error',
                message: 'Error Generating Draft',
                details: errorMessage,
            });
        });
    });

    it('resets the form and mutation state when "New Draft" is clicked', async () => {
        const mockDraft = {
            subject: 'Test Subject',
            body: 'Test Body',
            to: ['test@example.com'],
            cc: [],
        };
        // Set up the state to show a draft first
        useDraftEmailMock.mockReturnValue({
            mutate: mockMutate,
            data: { draft: mockDraft },
            error: null,
            isPending: false,
            reset: mockReset,
        });

        renderComponent();
        await waitFor(() => {
            expect(screen.getByText('Generated Draft')).toBeInTheDocument();
        });

        fireEvent.click(screen.getByText('New Draft'));

        await waitFor(() => {
            expect(mockReset).toHaveBeenCalled();
        });
    });
});
