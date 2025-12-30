import { Virtuoso } from 'react-virtuoso';
import type { Message } from '../../schemas/thread';
import MessageRow from './Message';

interface MessageListProps {
  messages: Message[];
}

export default function MessageList({ messages }: MessageListProps) {
  return (
    <Virtuoso
      style={{ height: '100%' }}
      data={messages}
      itemContent={(index, message) => {
        return <MessageRow message={message} />;
      }}
    />
  );
}
