
import { useState } from 'react';
import GlassCard from '../components/ui/GlassCard';
import { api } from '../lib/api';
import type { EmailDraft } from '../lib/api';
import { cn } from '../lib/utils';
import { Loader2, PenTool, Copy, Check, RefreshCw } from 'lucide-react';

const TONES = [
  { id: 'professional', label: 'Professional' },
  { id: 'friendly', label: 'Friendly' },
  { id: 'formal', label: 'Formal' },
  { id: 'concise', label: 'Concise' },
];

export default function DraftView() {
  const [instruction, setInstruction] = useState('');
  const [threadId, setThreadId] = useState('');
  const [tone, setTone] = useState('professional');
  const [isLoading, setIsLoading] = useState(false);
  const [draft, setDraft] = useState<EmailDraft | null>(null);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!instruction.trim() || isLoading) return;

    setIsLoading(true);
    setError(null);
    setDraft(null);

    try {
      const response = await api.draftEmail(instruction.trim(), threadId || undefined, tone);
      setDraft(response.draft);
    } catch (err) {
      console.error("Draft generation failed:", err);
      // SECURITY: Do not expose raw error messages to the user in production.
      if (import.meta.env.DEV) {
        setError(err instanceof Error ? err.message : 'Failed to generate draft');
      } else {
        setError('An unexpected error occurred. Please try again.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopy = async () => {
    if (!draft) return;
    const text = `Subject: ${draft.subject}\n\n${draft.body}`;
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleReset = () => {
    setDraft(null);
    setInstruction('');
    setThreadId('');
    setError(null);
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <header className="p-6 border-b border-white/5">
        <h1 className="text-3xl font-bold tracking-tight bg-gradient-to-r from-white to-white/60 bg-clip-text text-transparent">
          Draft Email
        </h1>
        <p className="text-white/40 mt-1">Generate professional email drafts using AI</p>
      </header>

      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-4xl mx-auto space-y-8">
          {/* Input Form */}
          {!draft && (
            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Instruction */}
              <div className="space-y-2">
                <label className="text-sm font-medium text-white/60">What should the email say?</label>
                <textarea
                  value={instruction}
                  onChange={(e) => setInstruction(e.target.value)}
                  placeholder="e.g., Reply to John thanking him for the meeting and propose next Tuesday at 2pm..."
                  className="w-full h-32 px-4 py-3 rounded-xl bg-white/5 border border-white/10 focus:border-blue-500/50 focus:outline-none focus:ring-2 focus:ring-blue-500/20 text-white placeholder-white/30 transition-all resize-none"
                  disabled={isLoading}
                />
              </div>

              {/* Thread Context (Optional) */}
              <div className="space-y-2">
                <label className="text-sm font-medium text-white/60">Thread ID (optional)</label>
                <input
                  type="text"
                  value={threadId}
                  onChange={(e) => setThreadId(e.target.value)}
                  placeholder="Leave empty to draft without context"
                  className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 focus:border-blue-500/50 focus:outline-none focus:ring-2 focus:ring-blue-500/20 text-white placeholder-white/30 transition-all font-mono text-sm"
                  disabled={isLoading}
                />
              </div>

              {/* Tone Selector */}
              <div className="space-y-2">
                <label className="text-sm font-medium text-white/60">Tone</label>
                <div className="flex flex-wrap gap-2">
                  {TONES.map((t) => (
                    <button
                      key={t.id}
                      type="button"
                      onClick={() => setTone(t.id)}
                      className={cn(
                        "px-4 py-2 rounded-lg text-sm font-medium transition-all",
                        tone === t.id
                          ? "bg-blue-600 text-white"
                          : "bg-white/5 text-white/60 hover:bg-white/10 hover:text-white"
                      )}
                    >
                      {t.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Submit */}
              <button
                type="submit"
                disabled={isLoading || !instruction.trim()}
                className="w-full py-4 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2 font-medium"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <PenTool className="w-5 h-5" />
                    Generate Draft
                  </>
                )}
              </button>
            </form>
          )}

          {/* Error */}
          {error && (
            <GlassCard className="p-4 border-red-500/30 bg-red-500/10">
              <p className="text-red-400">{error}</p>
            </GlassCard>
          )}

          {/* Generated Draft */}
          {draft && (
            <div className="space-y-4 animate-slide-up">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold">Generated Draft</h2>
                <div className="flex gap-2">
                  <button
                    onClick={handleCopy}
                    className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 text-sm transition-all"
                  >
                    {copied ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4" />}
                    {copied ? 'Copied!' : 'Copy'}
                  </button>
                  <button
                    onClick={handleReset}
                    className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 text-sm transition-all"
                  >
                    <RefreshCw className="w-4 h-4" />
                    New Draft
                  </button>
                </div>
              </div>

              <GlassCard className="p-6 space-y-4">
                {/* Subject */}
                <div className="space-y-1">
                  <span className="text-xs text-white/40 uppercase tracking-wider">Subject</span>
                  <p className="text-lg font-medium">{draft.subject}</p>
                </div>

                {/* Recipients */}
                {(draft.to?.length || draft.cc?.length) && (
                  <div className="flex gap-6 text-sm">
                    {draft.to?.length && (
                      <div>
                        <span className="text-white/40">To: </span>
                        <span>{draft.to.join(', ')}</span>
                      </div>
                    )}
                    {draft.cc?.length && (
                      <div>
                        <span className="text-white/40">CC: </span>
                        <span>{draft.cc.join(', ')}</span>
                      </div>
                    )}
                  </div>
                )}

                {/* Body */}
                <div className="pt-4 border-t border-white/10">
                  <pre className="whitespace-pre-wrap font-sans text-white/90 leading-relaxed">
                    {draft.body}
                  </pre>
                </div>
              </GlassCard>
            </div>
          )}

          {/* Empty State */}
          {!draft && !isLoading && !instruction && (
            <div className="text-center py-12">
              <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-purple-500/20 to-blue-500/20 flex items-center justify-center mx-auto mb-6">
                <PenTool className="w-10 h-10 text-purple-400" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Compose with AI</h3>
              <p className="text-white/40 max-w-md mx-auto">
                Describe what you want to say, and Cortex will draft a polished email for you.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
