
import { Virtuoso } from 'react-virtuoso';
import type { SearchResult } from '../../lib/api';
import GlassCard from '../ui/GlassCard';
import { Mail, FileText } from 'lucide-react';

interface MessageListProps {
  messages: SearchResult[];
}

export default function MessageList({ messages }: MessageListProps) {
  if (messages.length === 0) {
    return (
      <div className="text-center py-20">
        <FileText className="w-12 h-12 text-white/20 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-white/60 mb-2">No Content Found</h3>
        <p className="text-white/40">This thread may not exist or has not been indexed.</p>
      </div>
    );
  }

  return (
    <Virtuoso
      style={{ height: '100%' }}
      data={messages}
      itemContent={(_index, message) => (
        <div className="mb-4">
            <GlassCard className="p-4">
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1">
                  <p className="text-white/90 whitespace-pre-wrap">{message.content}</p>
                  <div className="mt-3 flex items-center gap-3 text-xs text-white/40">
                    {message.conversation_id && (
                      <span className="flex items-center gap-1">
                        <Mail className="w-3 h-3" />
                        {message.conversation_id.substring(0, 8)}...
                      </span>
                    )}
                    <span className="text-blue-400">{(message.score * 100).toFixed(0)}% relevance</span>
                  </div>
                </div>
              </div>
            </GlassCard>
        </div>
      )}
    />
  );
}
