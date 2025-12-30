import { Component, type ErrorInfo, type ReactNode } from 'react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
}

class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false
  };

  public static getDerivedStateFromError(_: Error): State {
    return { hasError: true };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Uncaught error:", error, errorInfo);
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center h-screen bg-gray-100 dark:bg-gray-900">
          <div className="p-8 bg-white rounded-lg shadow-md dark:bg-gray-800">
            <h1 className="mb-4 text-2xl font-bold text-center text-gray-800 dark:text-gray-200">Something went wrong.</h1>
            <p className="mb-6 text-center text-gray-600 dark:text-gray-400">We're sorry for the inconvenience. Please try again.</p>
            <button
              onClick={() => this.setState({ hasError: false })}
              className="px-4 py-2 font-bold text-white bg-blue-500 rounded hover:bg-blue-700"
            >
              Try again
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
