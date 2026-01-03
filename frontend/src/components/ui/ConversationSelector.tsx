import * as React from 'react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Check, ChevronDown, Loader2, Search, X } from 'lucide-react';
import { api } from '../../lib/api';
import type { ThreadListItem } from '../../lib/api';
import { cn } from '../../lib/utils';
import { Input } from './Input';

interface ConversationSelectorProps {
  value?: string;
  onValueChange: (value: string | undefined) => void;
  placeholder?: string;
  disabled?: boolean;
  className?: string;
}

export function ConversationSelector({
  value,
  onValueChange,
  placeholder = 'Select a conversation...',
  disabled = false,
  className,
}: ConversationSelectorProps) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState('');
  const [threads, setThreads] = useState<ThreadListItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedThread, setSelectedThread] = useState<ThreadListItem | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const fetchThreads = useCallback(async (query: string) => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    abortControllerRef.current = new AbortController();
    setLoading(true);

    try {
      const response = await api.listThreads(
        query || undefined,
        50,
        0,
        abortControllerRef.current.signal,
      );
      setThreads(response.threads);
    } catch (error) {
      if (error instanceof Error && error.name !== 'AbortError') {
        console.error('Failed to fetch threads:', error);
        setThreads([]);
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      if (open) {
        fetchThreads(search);
      }
    }, 300);

    return () => clearTimeout(timeoutId);
  }, [search, open, fetchThreads]);

  useEffect(() => {
    if (value && !selectedThread) {
      const thread = threads.find((t) => t.conversation_id === value);
      if (thread) {
        setSelectedThread(thread);
      }
    }
  }, [value, threads, selectedThread]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSelect = (thread: ThreadListItem) => {
    setSelectedThread(thread);
    onValueChange(thread.conversation_id);
    setOpen(false);
    setSearch('');
  };

  const handleClear = (e: React.MouseEvent) => {
    e.stopPropagation();
    setSelectedThread(null);
    onValueChange(undefined);
    setSearch('');
  };

  const getDisplayText = (thread: ThreadListItem) => {
    return thread.smart_subject || thread.subject || thread.folder_name;
  };

  const formatDate = (dateStr: string | null) => {
    if (!dateStr) return '';
    try {
      const date = new Date(dateStr);
      return date.toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
      });
    } catch {
      return '';
    }
  };

  return (
    <div ref={containerRef} className={cn('relative w-full', className)}>
      <button
        type="button"
        onClick={() => !disabled && setOpen(!open)}
        disabled={disabled}
        className={cn(
          'flex h-10 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm',
          'dark:bg-white/5 dark:border-white/10 dark:text-white',
          'focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
          'disabled:cursor-not-allowed disabled:opacity-50',
          className,
        )}
      >
        <span className={cn('truncate', !selectedThread && 'text-muted-foreground')}>
          {selectedThread ? getDisplayText(selectedThread) : placeholder}
        </span>
        <div className="flex items-center gap-1">
          {selectedThread && !disabled && (
            <X
              className="h-4 w-4 opacity-50 hover:opacity-100 cursor-pointer"
              onClick={handleClear}
            />
          )}
          <ChevronDown className={cn('h-4 w-4 opacity-50 transition-transform', open && 'rotate-180')} />
        </div>
      </button>

      {open && (
        <div
          className={cn(
            'absolute z-50 mt-1 w-full rounded-md border shadow-lg',
            'bg-gray-900 border-white/10 text-white',
            'animate-in fade-in-0 zoom-in-95',
          )}
        >
          <div className="flex items-center border-b border-white/10 px-3">
            <Search className="h-4 w-4 opacity-50" />
            <Input
              ref={inputRef}
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search conversations..."
              className="border-0 bg-transparent focus-visible:ring-0 focus-visible:ring-offset-0"
              autoFocus
            />
            {loading && <Loader2 className="h-4 w-4 animate-spin opacity-50" />}
          </div>

          <div className="max-h-64 overflow-y-auto p-1">
            {threads.length === 0 && !loading && (
              <div className="py-6 text-center text-sm text-muted-foreground">
                {search ? 'No conversations found' : 'No conversations available'}
              </div>
            )}

            {threads.map((thread) => (
              <button
                key={thread.conversation_id}
                type="button"
                onClick={() => handleSelect(thread)}
                className={cn(
                  'relative flex w-full cursor-pointer select-none flex-col items-start rounded-sm px-2 py-2 text-sm outline-none',
                  'hover:bg-white/10 focus:bg-white/10',
                  value === thread.conversation_id && 'bg-white/5',
                )}
              >
                <div className="flex w-full items-center justify-between gap-2">
                  <span className="truncate font-medium">{getDisplayText(thread)}</span>
                  {value === thread.conversation_id && (
                    <Check className="h-4 w-4 shrink-0 text-blue-400" />
                  )}
                </div>
                <div className="flex w-full items-center justify-between gap-2 text-xs text-muted-foreground">
                  <span className="truncate">{thread.participants_preview || 'No participants'}</span>
                  <span className="shrink-0">{formatDate(thread.latest_date)}</span>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
