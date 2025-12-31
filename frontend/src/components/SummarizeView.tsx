
import { useEffect } from 'react';
import { useSummarize } from '../hooks/useSummarize';
import GlassCard from './ui/GlassCard';
import { Alert, AlertDescription, AlertTitle } from './ui/Alert';
import { Skeleton } from './ui/Skeleton';
import { useToast } from '../contexts/toastContext';
import { Loader2, Sparkles, AlertTriangle } from 'lucide-react';
import type { ThreadSummary } from '../schemas/summarize';
import { Button } from './ui/Button';

interface SummarizeViewProps {
  threadId: string;
}

export default function SummarizeView({ threadId }: SummarizeViewProps) {
  const { addToast: toast } = useToast();
  const {
    summarize,
    data: summarizeData,
    isLoading: isSummarizing,
    error: summarizeError,
  } = useSummarize();

  useEffect(() => {
    if (summarizeError) {
      toast({
        type: 'error',
        message: 'Summarization Failed',
        details: 'An unexpected error occurred while generating the summary.',
      });
    }
  }, [summarizeError, toast]);

  const handleSummarize = () => {
    summarize({ threadId });
  };

  const summary = summarizeData?.summary;

  return (
    <div>
      <Button
        onClick={handleSummarize}
        disabled={isSummarizing}
        variant="glass"
        className="gap-2 text-sm font-medium"
      >
        {isSummarizing ? (
          <Loader2 className="w-4 h-4 animate-spin" />
        ) : (
          <Sparkles className="w-4 h-4" />
        )}
        Summarize
      </Button>

      <div className="mt-4">
        {isSummarizing && <SummarySkeleton />}
        {summarizeError && <SummaryError />}
        {summary && <SummaryDisplay summary={summary} />}
      </div>
    </div>
  );
}

function SummaryDisplay({ summary }: { summary: ThreadSummary }) {
  return (
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
            {summary.key_points.map((point: string, i: number) => (
              <li key={i} className="flex items-start gap-2 text-sm text-white/70">
                <span className="w-1 h-1 rounded-full bg-green-400 mt-2" />
                {point}
              </li>
            ))}
          </ul>
        </div>
      )}
    </GlassCard>
  );
}

function SummarySkeleton() {
  return (
    <GlassCard className="p-5">
      <h3 className="text-sm font-medium text-white/50 uppercase tracking-wider mb-3 flex items-center gap-2">
        <Sparkles className="w-4 h-4" />
        Generating Summary...
      </h3>
      <div className="space-y-3">
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-3/4" />
      </div>
    </GlassCard>
  );
}

function SummaryError() {
  return (
    <Alert variant="destructive">
      <AlertTriangle className="h-4 w-4" />
      <AlertTitle>Summarization Error</AlertTitle>
      <AlertDescription>
        There was an error generating the summary. Please try again.
      </AlertDescription>
    </Alert>
  );
}
