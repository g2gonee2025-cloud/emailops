import { useEffect } from 'react';
import { useNavigate, Outlet, NavLink, useLocation } from 'react-router-dom';
import ErrorBoundary from './components/ErrorBoundary';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { ToastProvider } from './components/ui/Toast';
import { cn } from './lib/utils';
import {
  LayoutDashboard,
  MessageSquare,
  Search,
  Upload,
  Settings,
  ShieldCheck,
  PenTool,
  LogOut,
  type LucideIcon
} from 'lucide-react';
import { AppRoutes } from './routes';

// Navigation Item Component
interface NavItemProps {
  icon: LucideIcon;
  label: string;
  to: string;
}

const NavItem = ({ icon: Icon, label, to }: NavItemProps) => {
  const location = useLocation();
  const isActive = location.pathname === to || (to === '/search' && location.pathname.startsWith('/thread'));

  return (
    <NavLink
      to={to}
      className={cn(
        "w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 group",
        isActive
          ? "bg-blue-600/20 text-blue-400 border border-blue-500/30"
          : "text-white/50 hover:text-white hover:bg-white/5 border border-transparent"
      )}
      title={label}
    >
      <Icon className={cn("w-5 h-5 transition-colors", isActive && "text-blue-400")} />
      <span className="font-medium text-sm">{label}</span>
      {isActive && (
        <div className="ml-auto w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
      )}
    </NavLink>
  );
};

// Protected layout wrapper
export function ProtectedLayout() {
  const { isAuthenticated, logout } = useAuth();
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

  // Navigation items
  const navItems = [
    { to: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { to: '/ask', label: 'Ask', icon: MessageSquare },
    { to: '/search', label: 'Search', icon: Search },
    { to: '/draft', label: 'Draft', icon: PenTool },
    { to: '/ingest', label: 'Ingestion', icon: Upload },
    { to: '/admin', label: 'Admin', icon: Settings },
  ];

  return (
    <div className="flex h-screen bg-[#0a0a0a] text-white overflow-hidden font-sans selection:bg-blue-500/30">
      {/* Ambient Background */}
      <div className="fixed inset-0 bg-gradient-radial from-blue-900/20 via-transparent to-transparent opacity-50 pointer-events-none" aria-hidden="true" />
      <div className="fixed inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-[0.03] pointer-events-none mix-blend-overlay" aria-hidden="true" />

      {/* Sidebar */}
      <aside className="w-64 flex flex-col z-20 bg-black/40 backdrop-blur-xl border-r border-white/5" role="navigation" aria-label="Main navigation">
        {/* Logo */}
        <div className="h-16 flex items-center px-6 border-b border-white/5 gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-lg shadow-blue-500/20" aria-hidden="true">
            <ShieldCheck className="text-white" size={18} />
          </div>
          <div>
            <span className="font-bold text-lg tracking-tight">Cortex</span>
            <span className="font-light text-lg opacity-60">UI</span>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-2" aria-label="Primary navigation">
          {navItems.map((item) => (
            <NavItem
              key={item.to}
              {...item}
            />
          ))}
        </nav>

        {/* Logout Button */}
        <div className="p-4 border-t border-white/5">
          <button
            onClick={logout}
            className="w-full flex items-center gap-3 px-4 py-3 rounded-xl text-white/50 hover:text-white hover:bg-white/5 transition-all"
          >
            <LogOut className="w-5 h-5" />
            <span className="font-medium text-sm">Logout</span>
          </button>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-white/5">
          <div className="flex items-center justify-between text-xs text-white/30">
            <span>EmailOps Cortex</span>
            <span className="font-mono">v3.1.0</span>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-hidden flex flex-col bg-gradient-to-br from-transparent via-transparent to-black/30" role="main" aria-label="Main content">
        <Outlet />
      </main>
    </div>
  );
}

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
