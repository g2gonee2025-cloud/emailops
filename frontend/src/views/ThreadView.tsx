import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import GlassCard from './ui/GlassCard';
import { api } from '../lib/api';
import type { SearchResult, ThreadSummary } from '../lib/api';
import {
  ArrowLeft,
  Mail,
  FileText,
  Loader2,
  Sparkles,
  MessageSquare,
  PenTool
} from 'lucide-react';

export function ThreadView() {
  const { id } = useParams<{ id: string }>();
  const threadId = id!;
  const navigate = useNavigate();
  const [chunks, setChunks] = useState<SearchResult[]>([]);
  const [summary, setSummary] = useState<ThreadSummary | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSummarizing, setIsSummarizing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadThreadContent = async () => {
      setIsLoading(true);
      setError(null);
      try {
        // Search for all chunks related to this thread
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

  const handleSummarize = async () => {
    setIsSummarizing(true);
    try {
      const response = await api.summarizeThread(threadId);
      setSummary(response.summary);
    } catch (error) {
      console.error('Failed to summarize:', error);
    } finally {
      setIsSummarizing(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <header className="p-6 border-b border-white/5">
        <div className="flex items-center gap-4 mb-4">
          <button
            onClick={() => navigate('/search')}
            className="p-2 rounded-lg hover:bg-white/5 transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div>
            <h1 className="text-2xl font-bold tracking-tight">Thread Details</h1>
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-2">
          <button
            onClick={() => navigate('/ask')}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-sm font-medium transition-all"
          >
            <MessageSquare className="w-4 h-4" />
            Ask About This
          </button>
          <button
            onClick={() => navigate('/draft')}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-purple-600 hover:bg-purple-500 text-sm font-medium transition-all"
          >
            <PenTool className="w-4 h-4" />
            Draft Reply
          </button>
          <button
            onClick={handleSummarize}
            disabled={isSummarizing}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 text-sm font-medium transition-all disabled:opacity-50"
          >
            {isSummarizing ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Sparkles className="w-4 h-4" />
            )}
            Summarize
          </button>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Summary (if generated) */}
        {summary && (
          <GlassCard className="p-5 border-green-500/30 bg-green-500/5">
            <h3 className="text-sm font-medium text-green-400 uppercase tracking-wider mb-3 flex items-center gap-2">
              <Sparkles className="w-4 h-4" />
              AI Summary
            </h3>
            <p className="text-white/90 leading-relaxed">{summary.summary}</p>

            {summary.key_points && summary.key_points.length > 0 && (
              <div className="mt-4 pt-4 border-t border-white/10">
                <h4 className="text-xs text-white/50 uppercase mb-2">Key Points</h4>
                <ul className="space-y-1">
                  {summary.key_points.map((point, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm text-white/70">
                      <span className="w-1 h-1 rounded-full bg-green-400 mt-2" />
                      {point}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </GlassCard>
        )}

        {/* Loading */}
        {isLoading && (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="w-8 h-8 animate-spin text-blue-400" />
          </div>
        )}

        {/* Thread Content */}
        {!isLoading && chunks.length > 0 && (
          <section>
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <FileText className="w-5 h-5 text-blue-400" />
              Thread Content ({chunks.length} chunks)
            </h2>
            <div className="space-y-4">
              {chunks.map((chunk, i) => (
                <GlassCard key={chunk.chunk_id || i} className="p-4">
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1">
                      <p className="text-white/90 whitespace-pre-wrap">{chunk.content}</p>
                      <div className="mt-3 flex items-center gap-3 text-xs text-white/40">
                        {chunk.conversation_id && (
                          <span className="flex items-center gap-1">
                            <Mail className="w-3 h-3" />
                            {chunk.conversation_id.substring(0, 8)}...
                          </span>
                        )}
                        <span className="text-blue-400">{(chunk.score * 100).toFixed(0)}% relevance</span>
                      </div>
                    </div>
                  </div>
                </GlassCard>
              ))}
            </div>
          </section>
        )}

        {/* Error State */}
        {error && (
          <div className="text-center py-20">
            <FileText className="w-12 h-12 text-red-500/50 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-red-500/80 mb-2">An Error Occurred</h3>
            <p className="text-white/40">{error}</p>
          </div>
        )}

        {/* Empty State */}
        {!isLoading && !error && chunks.length === 0 && (
          <div className="text-center py-20">
            <FileText className="w-12 h-12 text-white/20 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-white/60 mb-2">No Content Found</h3>
            <p className="text-white/40">This thread may not exist or has not been indexed.</p>
          </div>
        )}
      </div>
    </div>
  );
}
