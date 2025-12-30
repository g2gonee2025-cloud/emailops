import { memo } from 'react';
import type { Message } from '../../schemas/thread';

const Message = memo(({ message }: { message: Message }) => {
  return (
    <div className="p-4 border-b border-white/10">
      <div className="flex justify-between text-sm text-white/50">
        <div>From: {message.from}</div>
        <div>{new Date(message.date).toLocaleString()}</div>
      </div>
      <div className="mt-2 text-white/90">{message.subject}</div>
      <div className="mt-4 text-sm text-white/70 whitespace-pre-wrap">
        {message.body}
      </div>
    </div>
  );
});
Message.displayName = 'Message';

export default Message;
