import ErrorBoundary from './components/ErrorBoundary';
import { AuthProvider } from './contexts/AuthContext';
import { ToastProvider } from './contexts/toastContext';
import { AppRoutes } from './routes';

// Protected layout wrapper (imported directly in routes.tsx now)

function App() {
  return (
    <ToastProvider>
      <AuthProvider>
        <ErrorBoundary>
          <AppRoutes />
        </ErrorBoundary>
      </AuthProvider>
    </ToastProvider>
  );
}

export default App;
