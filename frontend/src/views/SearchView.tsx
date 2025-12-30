
import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import GlassCard from '../components/ui/GlassCard';
import { api } from '../lib/api';
import type { SearchResult } from '../lib/api';
import { cn } from '../lib/utils';
import { Search, Mail, FileText, X, ExternalLink } from 'lucide-react';
import { SkeletonCard } from '../components/ui/Skeleton';

export default function SearchView() {
  const navigate = useNavigate();
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [queryTime, setQueryTime] = useState<number | null>(null);
  const [totalCount, setTotalCount] = useState(0);
  const [hasSearched, setHasSearched] = useState(false);

  // Debounced search
  const performSearch = useCallback(async (searchQuery: string) => {
    if (!searchQuery.trim()) {
      setResults([]);
      setHasSearched(false);
      return;
    }

    setIsLoading(true);
    setHasSearched(true);

    try {
      const response = await api.search(searchQuery, 20);
      setResults(response.results);
      setQueryTime(response.query_time_ms);
      setTotalCount(response.total_count);
    } catch (error) {
      console.error('Search failed:', error);
      setResults([]);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Debounce effect
  useEffect(() => {
    const timer = setTimeout(() => {
      performSearch(query);
    }, 300);

    return () => clearTimeout(timer);
  }, [query, performSearch]);

  const handleClear = () => {
    setQuery('');
    setResults([]);
    setHasSearched(false);
  };

  const handleResultClick = (result: SearchResult) => {
    // Prefer thread_id, fall back to conversation_id
    const id = result.thread_id || result.conversation_id;
    if (id) {
      navigate(`/thread/${id}`);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <header className="p-6 border-b border-white/5 space-y-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight bg-gradient-to-r from-white to-white/60 bg-clip-text text-transparent">
            Search
          </h1>
          <p className="text-white/40 mt-1">Find conversations, emails, and documents</p>
        </div>

        {/* Search Input */}
        <div className="relative">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-white/30" aria-hidden="true" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search emails, conversations, claims..."
            className="w-full pl-12 pr-12 py-4 rounded-2xl bg-white/5 border border-white/10 focus:border-blue-500/50 focus:outline-none focus:ring-2 focus:ring-blue-500/20 text-white placeholder-white/30 transition-all text-lg"
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

        {/* Stats */}
        {hasSearched && !isLoading && (
          <div className="flex items-center gap-4 text-sm text-white/40">
            <span>{totalCount} results</span>
            {queryTime !== null && <span>• {queryTime.toFixed(0)}ms</span>}
          </div>
        )}
      </header>

      {/* Results */}
      <div className="flex-1 overflow-y-auto p-6">
        {isLoading && (
          <div className="space-y-4" role="status" aria-label="Loading search results">
            <SkeletonCard />
            <SkeletonCard />
            <SkeletonCard />
          </div>
        )}

        {!isLoading && !hasSearched && (
          <div className="h-full flex flex-col items-center justify-center text-center py-20">
            <div className="w-20 h-20 rounded-2xl bg-white/5 flex items-center justify-center mb-6">
              <Search className="w-10 h-10 text-white/20" />
            </div>
            <h2 className="text-xl font-semibold text-white/60 mb-2">Start Searching</h2>
            <p className="text-white/30 max-w-md">
              Enter a search query above to find relevant emails, conversations, and documents in your knowledge base.
            </p>
          </div>
        )}

        {!isLoading && hasSearched && results.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center text-center py-20">
            <div className="w-20 h-20 rounded-2xl bg-red-500/10 flex items-center justify-center mb-6">
              <FileText className="w-10 h-10 text-red-400/50" />
            </div>
            <h2 className="text-xl font-semibold text-white/60 mb-2">No Results Found</h2>
            <p className="text-white/30 max-w-md">
              Try adjusting your search terms or use different keywords.
            </p>
          </div>
        )}

        {!isLoading && results.length > 0 && (
          <div className="space-y-4">
            {results.map((result, i) => (
              <GlassCard
                key={result.chunk_id || i}
                className="p-5 hover:border-blue-500/30 transition-all cursor-pointer group"
                onClick={() => handleResultClick(result)}
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    {/* Content preview */}
                    <p className="text-white/90 line-clamp-3 mb-3">{result.content}</p>

                    {/* Metadata */}
                    <div className="flex items-center gap-4 text-xs text-white/40">
                      {result.conversation_id && (
                        <span className="flex items-center gap-1">
                          <Mail className="w-3 h-3" />
                          {result.conversation_id.substring(0, 8)}...
                        </span>
                      )}
                      {result.thread_id && (
                        <span className="flex items-center gap-1">
                          <FileText className="w-3 h-3" />
                          Thread: {result.thread_id.substring(0, 8)}...
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Score & Link */}
                  <div className="flex flex-col items-end gap-2">
                    <span className={cn(
                      "px-2 py-1 rounded text-xs font-medium",
                      result.score >= 0.8 ? "bg-green-500/20 text-green-400" :
                      result.score >= 0.5 ? "bg-yellow-500/20 text-yellow-400" :
                      "bg-white/10 text-white/60"
                    )}>
                      {(result.score * 100).toFixed(0)}% match
                    </span>
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
        )}
      </div>
    </div>
  );
}
