import { useEffect } from 'react';
import { useNavigate, Outlet } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { Sidebar } from './Sidebar';

export default function Layout() {
  const { isAuthenticated } = useAuth();
  const navigate = useNavigate();

  // Redirect if not authenticated
  useEffect(() => {
    if (!isAuthenticated) {
      navigate('/login');
    }
  }, [isAuthenticated, navigate]);

  if (!isAuthenticated) {
    return null; // Will redirect via useEffect
  }

  return (
    <div className="flex min-h-screen text-white overflow-hidden font-sans selection:bg-emerald-500/30">
      <Sidebar />

      {/* Main Content */}
      <main className="flex-1 overflow-hidden flex flex-col" role="main" aria-label="Main content">
        <Outlet />
      </main>
    </div>
  );
}
