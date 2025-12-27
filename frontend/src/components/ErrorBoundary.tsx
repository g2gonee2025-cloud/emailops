import { Component, type ErrorInfo, type ReactNode } from 'react';
import { logger } from '../lib/logger';
import GlassCard from './ui/GlassCard';
import { AlertTriangle } from 'lucide-react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorId: string | null;
}

class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
    errorId: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    const errorId = crypto.randomUUID();
    return { hasError: true, error, errorId };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    logger.error('Uncaught error in component tree:', {
      error,
      errorInfo,
      errorId: this.state.errorId,
    });
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center p-4 bg-background text-foreground">
          <GlassCard
            className="max-w-md w-full p-8 flex flex-col items-center text-center space-y-4"
            intensity="high"
          >
            <div className="p-3 bg-red-500/10 rounded-full text-red-500">
              <AlertTriangle size={48} />
            </div>
            <h1 className="text-2xl font-bold font-display">
              System Malfunction
            </h1>
            <p className="text-muted-foreground">
              An unexpected anomaly has occurred. Our systems have logged this
              event for analysis.
              {this.state.errorId && (
                <span className="block mt-2 text-xs">
                  Error ID: {this.state.errorId}
                </span>
              )}
            </p>

            {import.meta.env.DEV && (
              <div className="w-full bg-black/20 p-4 rounded text-left overflow-auto max-h-40 text-xs font-mono text-red-400">
                {this.state.error?.toString()}
              </div>
            )}
            <button
              onClick={() => window.location.reload()}
              className="px-6 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors font-medium"
            >
              Reboot System
            </button>
          </GlassCard>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
