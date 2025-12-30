import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { ChatWindow } from './ChatWindow';

describe('ChatWindow', () => {
  const messages = [
    { role: 'user' as const, content: 'Hello' },
    { role: 'assistant' as const, content: 'Hi there!' },
  ];

  it('renders messages correctly', () => {
    render(<ChatWindow messages={messages} isStreaming={false} />);
    expect(screen.getByText('Hello')).toBeInTheDocument();
    expect(screen.getByText('Hi there!')).toBeInTheDocument();
  });

  it('shows streaming placeholder when isStreaming is true', () => {
    render(<ChatWindow messages={messages} isStreaming={true} />);
    expect(screen.getByText('Thinking...')).toBeInTheDocument();
  });

  it('does not show streaming placeholder when isStreaming is false', () => {
    render(<ChatWindow messages={messages} isStreaming={false} />);
    expect(screen.queryByText('Thinking...')).not.toBeInTheDocument();
  });
});
