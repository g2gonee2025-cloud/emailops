
import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import MessageList from './MessageList';
import type { SearchResult } from '../../lib/api';
import { AllTheProviders } from '../../tests/testUtils';

vi.mock('react-virtuoso', () => ({
  Virtuoso: ({ data, itemContent }: { data: SearchResult[], itemContent: (index: number, item: SearchResult) => React.ReactNode }) => (
    <div>
      {data.map((item, index) => (
        <div key={index}>{itemContent(index, item)}</div>
      ))}
    </div>
  ),
}));

const mockMessages: SearchResult[] = [
  {
    chunk_id: '1',
    conversation_id: 'conv1',
    content: 'This is the first message.',
    score: 0.95,
    role: 'user',
    timestamp: '2024-01-01T12:00:00Z',
    sender: 'test@example.com',
    recipients: ['recipient@example.com'],
    subject: 'Test Subject 1'
  },
  {
    chunk_id: '2',
    conversation_id: 'conv2',
    content: 'This is the second message.',
    score: 0.85,
    role: 'user',
    timestamp: '2024-01-01T12:01:00Z',
    sender: 'test2@example.com',
    recipients: ['recipient2@example.com'],
    subject: 'Test Subject 2'
  },
];

describe('MessageList', () => {
  it('renders messages correctly', () => {
    render(
      <AllTheProviders>
        <MessageList messages={mockMessages} />
      </AllTheProviders>
    );

    expect(screen.getByText('This is the first message.')).toBeInTheDocument();
    expect(screen.getByText('This is the second message.')).toBeInTheDocument();
    expect(screen.getByText('95% relevance')).toBeInTheDocument();
    expect(screen.getByText('85% relevance')).toBeInTheDocument();
  });

  it('renders no content message when there are no messages', () => {
    render(
      <AllTheProviders>
        <MessageList messages={[]} />
      </AllTheProviders>
    );

    expect(screen.getByText('No Content Found')).toBeInTheDocument();
  });
});
