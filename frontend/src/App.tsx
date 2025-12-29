import { useEffect } from 'react';
import { Routes, Route, Navigate, useNavigate, useLocation } from 'react-router-dom';
import ErrorBoundary from './components/ErrorBoundary';
import { DashboardView } from './components/DashboardView';
import { AskView } from './components/AskView';
import { SearchView } from './components/SearchView';
import { IngestionView } from './components/IngestionView';
import { DraftView } from './components/DraftView';
import { SummarizeView } from './components/SummarizeView';
import { ThreadView } from './components/ThreadView';
import { AdminDashboard } from './components/AdminDashboard';
import { LoginView } from './components/LoginView';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { ToastProvider } from './components/ui/Toast';
import { Sidebar } from './components/Sidebar';

// Protected layout wrapper
function ProtectedLayout() {
  const { isAuthenticated } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  // Redirect if not authenticated
  useEffect(() => {
    if (!isAuthenticated) {
      navigate('/login', { state: { from: location } });
    }
  }, [isAuthenticated, navigate, location]);

  if (!isAuthenticated) {
    return null; // Will redirect via useEffect
  }

  return (
    <div className="flex h-screen bg-[#0a0a0a] text-white overflow-hidden font-sans selection:bg-blue-500/30">
      {/* Ambient Background */}
      <div className="fixed inset-0 bg-gradient-radial from-blue-900/20 via-transparent to-transparent opacity-50 pointer-events-none" aria-hidden="true" />
      <div className="fixed inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-[0.03] pointer-events-none mix-blend-overlay" aria-hidden="true" />

      <Sidebar />

      {/* Main Content */}
      <main className="flex-1 overflow-hidden flex flex-col bg-gradient-to-br from-transparent via-transparent to-black/30" role="main" aria-label="Main content">
        <Routes>
          <Route path="dashboard" element={<DashboardView />} />
          <Route path="ask" element={<AskView />} />
          <Route path="search" element={<SearchView />} />
          <Route path="thread/:threadId" element={<ThreadView />} />
          <Route path="draft" element={<DraftView />} />
          <Route path="summarize" element={<SummarizeView />} />
          <Route path="ingestion" element={<IngestionView />} />
          <Route path="admin" element={<AdminDashboard />} />
          <Route index element={<Navigate to="dashboard" replace />} />
        </Routes>
      </main>
    </div>
  );
}

function App() {
  return (
    <ToastProvider>
      <AuthProvider>
        <ErrorBoundary>
          <Routes>
            <Route path="/login" element={<LoginView />} />
            <Route path="/*" element={<ProtectedLayout />} />
          </Routes>
        </ErrorBoundary>
      </AuthProvider>
    </ToastProvider>
  );
}

export default App;
