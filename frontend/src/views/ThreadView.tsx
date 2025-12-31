
import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { api } from '../lib/api';
import type { SearchResult } from '../lib/api';
import {
  ArrowLeft,
  FileText,
  Loader2,
  MessageSquare,
  PenTool
} from 'lucide-react';
import SummarizeView from '../components/SummarizeView';
import MessageList from '../components/thread/MessageList';
import { Button } from '../components/ui/Button';

export default function ThreadView() {
  const { id } = useParams<{ id: string }>();
  const threadId = id!;
  const navigate = useNavigate();
  const [chunks, setChunks] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadThreadContent = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await api.search(`thread:${threadId}`, 50);
        setChunks(response.results);
      } catch (error) {
        console.error('Failed to load thread:', error);
        setError('Failed to load thread content. Please try again later.');
      } finally {
        setIsLoading(false);
      }
    };
    loadThreadContent();
  }, [threadId]);

  return (
    <div className="flex flex-col h-full">
      <header className="p-6 border-b border-white/5">
        <div className="flex items-center gap-4 mb-4">
          <Button
            onClick={() => navigate('/search')}
            variant="ghost"
            size="icon"
          >
            <ArrowLeft className="w-5 h-5" />
          </Button>
          <div>
            <h1 className="text-2xl font-display font-semibold tracking-tight">Thread Details</h1>
          </div>
        </div>
        <div className="flex gap-2">
          <Button
            onClick={() => navigate('/ask')}
            variant="glass"
            className="gap-2 text-sm font-medium"
          >
            <MessageSquare className="w-4 h-4" />
            Ask About This
          </Button>
          <Button
            onClick={() => navigate('/draft')}
            className="gap-2 text-sm font-medium"
          >
            <PenTool className="w-4 h-4" />
            Draft Reply
          </Button>
        </div>
      </header>

      <div className="flex-1 flex flex-col min-h-0">
        <div className="p-6 space-y-6">
            <SummarizeView threadId={threadId} />
        </div>

        {isLoading && (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="w-8 h-8 animate-spin text-blue-400" />
          </div>
        )}

        {!isLoading && (
          <div className="flex-1 flex flex-col min-h-0 px-6 pb-6">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <FileText className="w-5 h-5 text-blue-400" />
              Thread Content ({chunks.length} chunks)
            </h2>
            <div className="flex-1 min-h-0">
                <MessageList messages={chunks} />
            </div>
          </div>
        )}

        {error && (
          <div className="text-center py-20 px-6">
            <FileText className="w-12 h-12 text-red-500/50 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-red-500/80 mb-2">An Error Occurred</h3>
            <p className="text-white/40">{error}</p>
          </div>
        )}

      </div>
    </div>
  );
}
