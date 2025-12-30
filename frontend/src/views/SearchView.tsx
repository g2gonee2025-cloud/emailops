
import { useState, useMemo, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Search,
  Mail,
  FileText,
  X,
  ExternalLink,
  AlertTriangle,
  FileQuestion,
} from 'lucide-react';

import { useDebounce } from '../hooks/useDebounce';
import { useSearch } from '../hooks/useSearch';
import { type SearchResult, SearchResponseSchema } from '../schemas/search';
import { logger } from '../lib/logger';

import { Input } from '../components/ui/Input';
import { Skeleton } from '../components/ui/Skeleton';
import GlassCard from '../components/ui/GlassCard';
import { Badge } from '../components/ui/Badge';
import { Alert, AlertDescription, AlertTitle } from '../components/ui/Alert';
import { useToast } from '../contexts/toastContext';

const SKELETON_COUNT = 3;

export default function SearchView() {
  const navigate = useNavigate();
  const { addToast } = useToast();
  const [query, setQuery] = useState('');
  const debouncedQuery = useDebounce(query, 300);

  const { data, isLoading, isError, error } = useSearch({
    query: debouncedQuery,
    k: 20,
  });

  // Display a toast notification on error
  useEffect(() => {
    if (isError) {
      logger.error('Search query failed', { error });
      addToast({
        type: 'error',
        message: 'Search Failed',
        details: error instanceof Error ? error.message : 'An unknown error occurred.',
      });
    }
  }, [isError, error, addToast]);

  // Safely parse and memoize the API response
  const validatedData = useMemo(() => {
    if (!data) return null;
    const parseResult = SearchResponseSchema.safeParse(data);
    if (!parseResult.success) {
      logger.warn('Invalid search response schema', {
        errors: parseResult.error.flatten(),
        rawData: data,
      });
      return null;
    }
    return parseResult.data;
  }, [data]);

  const handleClear = () => {
    setQuery('');
  };

  const handleResultClick = (result: SearchResult) => {
    const id = result.thread_id || result.conversation_id;
    if (id) {
      navigate(`/thread/${id}`);
    }
  };

  const renderScoreBadge = (score: number) => {
    const scorePercentage = (score * 100).toFixed(0);
    let variant: 'default' | 'secondary' | 'destructive' = 'default';
    if (score >= 0.8) {
      variant = 'default'; // default is green-ish
    } else if (score >= 0.5) {
      variant = 'secondary'; // secondary is yellow-ish
    } else {
      variant = 'destructive'; // destructive is red-ish
    }

    return <Badge variant={variant}>{scorePercentage}% match</Badge>;
  };

  const renderContent = () => {
    if (isLoading) {
      return (
        <div className="space-y-4" role="status" aria-label="Loading search results">
          {Array.from({ length: SKELETON_COUNT }).map((_, i) => (
            <div key={i} className="p-5 border rounded-lg bg-white/5 border-white/10">
              <Skeleton className="h-4 w-3/4 mb-3" />
              <Skeleton className="h-3 w-1/2" />
            </div>
          ))}
        </div>
      );
    }

    if (isError) {
      return (
        <Alert variant="destructive" className="max-w-xl mx-auto">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>
            There was a problem with your search. Please try again later.
          </AlertDescription>
        </Alert>
      );
    }

    if (!debouncedQuery.trim()) {
      return (
        <div className="text-center py-20">
          <div className="w-20 h-20 rounded-2xl bg-white/5 flex items-center justify-center mb-6 mx-auto">
            <Search className="w-10 h-10 text-white/20" />
          </div>
          <h2 className="text-xl font-semibold text-white/60 mb-2">Start Searching</h2>
          <p className="text-white/30 max-w-md mx-auto">
            Find relevant emails, conversations, and documents in your knowledge base.
          </p>
        </div>
      );
    }

    if (!validatedData || validatedData.results.length === 0) {
      return (
        <Alert className="max-w-xl mx-auto">
          <FileQuestion className="h-4 w-4" />
          <AlertTitle>No Results Found</AlertTitle>
          <AlertDescription>
            Try adjusting your search terms or use different keywords.
          </AlertDescription>
        </Alert>
      );
    }

    return (
      <div className="space-y-4">
        {validatedData.results.map((result) => (
          <GlassCard
            key={result.chunk_id}
            className="p-5 hover:border-blue-500/30 transition-all cursor-pointer group"
            onClick={() => handleResultClick(result)}
          >
            <div className="flex items-start justify-between gap-4">
              <div className="flex-1 min-w-0">
                <p className="text-white/90 line-clamp-3 mb-3">{result.content}</p>
                <div className="flex items-center gap-4 text-xs text-white/40">
                  {result.conversation_id && (
                    <span className="flex items-center gap-1.5">
                      <Mail className="w-3 h-3" />
                      {result.conversation_id.substring(0, 8)}...
                    </span>
                  )}
                  {result.thread_id && (
                    <span className="flex items-center gap-1.5">
                      <FileText className="w-3 h-3" />
                      Thread: {result.thread_id.substring(0, 8)}...
                    </span>
                  )}
                </div>
              </div>
              <div className="flex flex-col items-end gap-2 shrink-0">
                {renderScoreBadge(result.score)}
                {(result.thread_id || result.conversation_id) && (
                  <span className="text-xs text-blue-400 opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-1">
                    <ExternalLink className="w-3 h-3" /> View
                  </span>
                )}
              </div>
            </div>
          </GlassCard>
        ))}
      </div>
    );
  };

  return (
    <div className="flex flex-col h-full">
      <header className="p-6 border-b border-white/5 space-y-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight bg-gradient-to-r from-white to-white/60 bg-clip-text text-transparent">
            Search
          </h1>
          <p className="text-white/40 mt-1">Find conversations, emails, and documents</p>
        </div>

        <div className="relative">
          <Search
            className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-white/30"
            aria-hidden="true"
          />
          <Input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search emails, conversations, claims..."
            className="pl-12 pr-12 py-4 h-auto text-lg"
            aria-label="Search query"
          />
          {query && (
            <button
              onClick={handleClear}
              className="absolute right-4 top-1/2 -translate-y-1/2 p-1 rounded-lg hover:bg-white/10 transition-colors"
              aria-label="Clear search"
            >
              <X className="w-5 h-5 text-white/50" />
            </button>
          )}
        </div>

        {debouncedQuery && !isLoading && validatedData && (
          <div className="flex items-center gap-4 text-sm text-white/40">
            <span>{validatedData.total_count} results</span>
            <span>• {validatedData.query_time_ms.toFixed(0)}ms</span>
          </div>
        )}
      </header>

      <div className="flex-1 overflow-y-auto p-6">{renderContent()}</div>
    </div>
  );
}
