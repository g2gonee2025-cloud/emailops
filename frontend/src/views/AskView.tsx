import { useState, useRef, useEffect } from 'react';
import GlassCard from '../components/ui/GlassCard';
import { api } from '../lib/api';
import type { ChatMessage as ApiChatMessage, Answer } from '../lib/api';
import { cn } from '../lib/utils';
import { Send, Bot, User, Loader2, Sparkles, FileText } from 'lucide-react';
import { Input } from '../components/ui/Input';
import { Button } from '../components/ui/Button';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: Answer['sources'];
  timestamp: Date;
}

export default function AskView() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      // Build chat history for API
      const apiMessages: ApiChatMessage[] = messages.map(m => ({
        role: m.role,
        content: m.content,
      }));
      apiMessages.push({ role: 'user', content: userMessage.content });

      const response = await api.chat(apiMessages);

      const assistantMessage: Message = {
        role: 'assistant',
        content: response.reply,
        sources: response.answer?.sources,
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage: Message = {
        role: 'assistant',
        content: `Sorry, I encountered an error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <header className="p-6 border-b border-white/5">
        <h1 className="text-3xl font-display font-semibold tracking-tight bg-gradient-to-r from-white to-white/60 bg-clip-text text-transparent">
          Ask Cortex
        </h1>
        <p className="text-white/40 mt-1">Query your knowledge base with natural language</p>
      </header>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center text-center py-20">
            <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 flex items-center justify-center mb-6">
              <Sparkles className="w-10 h-10 text-emerald-300" />
            </div>
            <h2 className="text-2xl font-display font-semibold mb-2">Welcome to Cortex AI</h2>
            <p className="text-white/40 max-w-md">
              Ask questions about your emails, conversations, and documents. I'll search the knowledge base and provide answers with sources.
            </p>
            <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-4 max-w-2xl">
              {[
                "What are the open claims?",
                "Summarize recent finance emails",
                "Who contacted me about Project Alpha?",
                "Find emails about the Q4 budget",
              ].map((suggestion, i) => (
                <Button
                  key={i}
                  onClick={() => setInput(suggestion)}
                  variant="glass"
                  className="h-auto p-4 text-left justify-start border-white/10 hover:border-emerald-400/40 text-sm text-white/70 whitespace-normal"
                >
                  {suggestion}
                </Button>
              ))}
            </div>
          </div>
        )}

        {messages.map((message, i) => (
          <div
            key={i}
            className={cn(
              "flex gap-4 animate-slide-up",
              message.role === 'user' ? "justify-end" : "justify-start"
            )}
          >
            {message.role === 'assistant' && (
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-400 to-cyan-500 flex items-center justify-center flex-shrink-0">
                <Bot className="w-5 h-5 text-white" />
              </div>
            )}

            <div className={cn(
              "max-w-[70%] space-y-3",
              message.role === 'user' ? "items-end" : "items-start"
            )}>
              <GlassCard className={cn(
                "p-4",
                message.role === 'user'
                  ? "bg-emerald-500/15 border-emerald-400/30"
                  : "bg-white/5"
              )}>
                <p className="text-white/90 whitespace-pre-wrap">{message.content}</p>
              </GlassCard>

              {/* Sources */}
              {message.sources && message.sources.length > 0 && (
                <div className="space-y-2">
                  <span className="text-xs text-white/40 uppercase tracking-wider">Sources</span>
                  <div className="flex flex-wrap gap-2">
                    {message.sources.slice(0, 3).map((source, si) => (
                      <div
                        key={si}
                        className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 text-xs"
                      >
                        <FileText className="w-3 h-3 text-emerald-300" />
                        <span className="text-white/60 truncate max-w-[200px]">
                          {source.content.substring(0, 50)}...
                        </span>
                        <span className="text-emerald-300">{(source.score * 100).toFixed(0)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {message.role === 'user' && (
              <div className="w-10 h-10 rounded-xl bg-white/10 flex items-center justify-center flex-shrink-0">
                <User className="w-5 h-5 text-white/70" />
              </div>
            )}
          </div>
        ))}

        {isLoading && (
          <div className="flex gap-4 animate-slide-up">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-400 to-cyan-500 flex items-center justify-center">
              <Bot className="w-5 h-5 text-white" />
            </div>
            <GlassCard className="p-4">
              <div className="flex items-center gap-3">
                <Loader2 className="w-4 h-4 animate-spin text-emerald-300" />
                <span className="text-white/60">Thinking...</span>
              </div>
            </GlassCard>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-6 border-t border-white/5">
        <form onSubmit={handleSubmit} className="flex gap-4">
          <Input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask anything about your emails..."
            className="flex-1 h-12 px-4 text-base bg-white/5 border-white/10 focus:border-emerald-400/50 focus:ring-emerald-500/30 text-white placeholder-white/30"
            disabled={isLoading}
          />
          <Button
            type="submit"
            disabled={isLoading || !input.trim()}
            size="lg"
            className="h-12 px-5 gap-2 font-medium"
          >
            <Send className="w-5 h-5" />
            <span className="hidden md:inline">Send</span>
          </Button>
        </form>
      </div>
    </div>
  );
}
