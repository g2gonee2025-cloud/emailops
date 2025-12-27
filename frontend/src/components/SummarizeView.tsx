import { useState } from 'react';
import GlassCard from './ui/GlassCard';
import { api } from '../lib/api';
import type { ThreadSummary } from '../lib/api';
import { FileText, Loader2, ListChecks, Sparkles, Copy, Check } from 'lucide-react';

export function SummarizeView() {
  const [threadId, setThreadId] = useState('');
  const [maxLength, setMaxLength] = useState(500);
  const [isLoading, setIsLoading] = useState(false);
  const [summary, setSummary] = useState<ThreadSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!threadId.trim() || isLoading) return;

    setIsLoading(true);
    setError(null);
    setSummary(null);

    try {
      const response = await api.summarizeThread(threadId.trim(), maxLength);
      setSummary(response.summary);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to summarize thread');
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopy = async () => {
    if (!summary) return;
    let text = summary.summary;
    if (summary.key_points?.length) {
      text += '\n\nKey Points:\n' + summary.key_points.map(p => `• ${p}`).join('\n');
    }
    if (summary.action_items?.length) {
      text += '\n\nAction Items:\n' + summary.action_items.map(a => `☐ ${a}`).join('\n');
    }
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <header className="p-6 border-b border-white/5">
        <h1 className="text-3xl font-bold tracking-tight bg-gradient-to-r from-white to-white/60 bg-clip-text text-transparent">
          Summarize Thread
        </h1>
        <p className="text-white/40 mt-1">Get AI-generated summaries of email threads</p>
      </header>

      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-4xl mx-auto space-y-8">
          {/* Input Form */}
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Thread ID */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-white/60">Thread ID</label>
              <input
                type="text"
                value={threadId}
                onChange={(e) => setThreadId(e.target.value)}
                placeholder="Enter the thread ID to summarize"
                className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 focus:border-blue-500/50 focus:outline-none focus:ring-2 focus:ring-blue-500/20 text-white placeholder-white/30 transition-all font-mono"
                disabled={isLoading}
              />
            </div>

            {/* Max Length */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-white/60">Max Length: {maxLength} words</label>
              <input
                type="range"
                min={100}
                max={1000}
                step={50}
                value={maxLength}
                onChange={(e) => setMaxLength(Number(e.target.value))}
                className="w-full"
                disabled={isLoading}
              />
              <div className="flex justify-between text-xs text-white/30">
                <span>Brief</span>
                <span>Detailed</span>
              </div>
            </div>

            {/* Submit */}
            <button
              type="submit"
              disabled={isLoading || !threadId.trim()}
              className="w-full py-4 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2 font-medium"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Summarizing...
                </>
              ) : (
                <>
                  <Sparkles className="w-5 h-5" />
                  Summarize
                </>
              )}
            </button>
          </form>

          {/* Error */}
          {error && (
            <GlassCard className="p-4 border-red-500/30 bg-red-500/10">
              <p className="text-red-400">{error}</p>
            </GlassCard>
          )}

          {/* Summary Result */}
          {summary && (
            <div className="space-y-4 animate-slide-up">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold">Summary</h2>
                <button
                  onClick={handleCopy}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 text-sm transition-all"
                >
                  {copied ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4" />}
                  {copied ? 'Copied!' : 'Copy'}
                </button>
              </div>

              <GlassCard className="p-6 space-y-6">
                {/* Main Summary */}
                <div>
                  <p className="text-white/90 leading-relaxed">{summary.summary}</p>
                </div>

                {/* Key Points */}
                {summary.key_points && summary.key_points.length > 0 && (
                  <div className="space-y-3">
                    <h3 className="text-sm font-medium text-white/60 uppercase tracking-wider flex items-center gap-2">
                      <FileText className="w-4 h-4" />
                      Key Points
                    </h3>
                    <ul className="space-y-2">
                      {summary.key_points.map((point, i) => (
                        <li key={i} className="flex items-start gap-3">
                          <span className="w-1.5 h-1.5 rounded-full bg-blue-400 mt-2 flex-shrink-0" />
                          <span className="text-white/80">{point}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Action Items */}
                {summary.action_items && summary.action_items.length > 0 && (
                  <div className="space-y-3">
                    <h3 className="text-sm font-medium text-white/60 uppercase tracking-wider flex items-center gap-2">
                      <ListChecks className="w-4 h-4" />
                      Action Items
                    </h3>
                    <ul className="space-y-2">
                      {summary.action_items.map((item, i) => (
                        <li key={i} className="flex items-start gap-3">
                          <span className="w-4 h-4 rounded border border-white/20 mt-0.5 flex-shrink-0" />
                          <span className="text-white/80">{item}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </GlassCard>
            </div>
          )}

          {/* Empty State */}
          {!summary && !isLoading && !threadId && (
            <div className="text-center py-12">
              <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-green-500/20 to-blue-500/20 flex items-center justify-center mx-auto mb-6">
                <FileText className="w-10 h-10 text-green-400" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Thread Summarization</h3>
              <p className="text-white/40 max-w-md mx-auto">
                Enter a thread ID to get an AI-generated summary with key points and action items.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
