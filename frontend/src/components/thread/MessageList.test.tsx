import { render, screen, within } from '@testing-library/react';
import MessageList from './MessageList';
import type { Message } from '../../schemas/thread';
import { AllTheProviders } from '../../tests/testUtils';

// Mock react-virtuoso to render all items for testing
vi.mock('react-virtuoso', async () => {
    const Virtuoso = ({ data, itemContent }: { data: Message[], itemContent: (index: number, item: Message) => React.ReactNode }) => {
        return (
            <div data-testid="mock-virtuoso">
                {data.map((item, index) => (
                    <div key={item.id || index}>
                        {itemContent(index, item)}
                    </div>
                ))}
            </div>
        );
    };
    return { Virtuoso };
});


const mockMessages: Message[] = [
  {
    id: '1',
    from: 'sender1@example.com',
    to: ['receiver@example.com'],
    subject: 'Subject 1',
    body: 'Body 1',
    date: new Date('2024-01-01T10:00:00Z').toISOString(),
  },
  {
    id: '2',
    from: 'sender2@example.com',
    to: ['receiver@example.com'],
    subject: 'Subject 2',
    body: 'Body 2',
    date: new Date('2024-01-01T11:00:00Z').toISOString(),
  },
];

describe('MessageList', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders a list of messages', () => {
    render(
        <AllTheProviders>
            <MessageList messages={mockMessages} />
        </AllTheProviders>
    );

    const list = screen.getByTestId('mock-virtuoso');

    expect(within(list).getByText('Subject 1')).toBeInTheDocument();
    expect(within(list).getByText('From: sender1@example.com')).toBeInTheDocument();

    expect(within(list).getByText('Subject 2')).toBeInTheDocument();
    expect(within(list).getByText('From: sender2@example.com')).toBeInTheDocument();
  });
});
