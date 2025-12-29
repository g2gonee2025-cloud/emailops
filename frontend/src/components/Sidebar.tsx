import { useState } from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  MessageSquare,
  Search,
  Upload,
  Settings,
  ShieldCheck,
  PenTool,
  FileText,
  LogOut,
  ChevronsLeft,
  ChevronsRight,
  type LucideIcon,
} from 'lucide-react';
import { cn } from '../lib/utils';
import { useAuth } from '../contexts/AuthContext';

interface NavItemProps {
  icon: LucideIcon;
  label: string;
  to: string;
  isCollapsed: boolean;
}

const NavItem = ({ icon: Icon, label, to, isCollapsed }: NavItemProps) => (
  <NavLink
    to={to}
    className={({ isActive }) =>
      cn(
        'w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 group',
        isActive
          ? 'bg-blue-600/20 text-blue-400 border border-blue-500/30'
          : 'text-white/50 hover:text-white hover:bg-white/5 border border-transparent',
        isCollapsed && 'justify-center',
      )
    }
    aria-label={label}
  >
    <Icon className={cn('w-5 h-5 transition-colors')} />
    {!isCollapsed && <span className="font-medium text-sm">{label}</span>}
    {!isCollapsed && (
      <div className="ml-auto w-1.5 h-1.5 rounded-full bg-transparent group-hover:bg-blue-400 transition-colors" />
    )}
  </NavLink>
);

export const Sidebar = () => {
  const { logout } = useAuth();
  const [isCollapsed, setIsCollapsed] = useState(false);

  const navItems = [
    { to: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { to: '/ask', label: 'Ask', icon: MessageSquare },
    { to: '/search', label: 'Search', icon: Search },
    { to: '/draft', label: 'Draft', icon: PenTool },
    { to: '/summarize', label: 'Summarize', icon: FileText },
    { to: '/ingestion', label: 'Ingestion', icon: Upload },
    { to: '/admin', label: 'Admin', icon: Settings },
  ];

  return (
    <aside
      className={cn(
        'flex flex-col z-20 bg-black/40 backdrop-blur-xl border-r border-white/5 transition-all duration-300',
        isCollapsed ? 'w-20' : 'w-64',
      )}
      role="navigation"
      aria-label="Main navigation"
    >
      <div className="h-16 flex items-center px-6 border-b border-white/5 gap-3">
        <div
          className="w-9 h-9 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-lg shadow-blue-500/20"
          aria-hidden="true"
        >
          <ShieldCheck className="text-white" size={18} />
        </div>
        {!isCollapsed && (
          <div>
            <span className="font-bold text-lg tracking-tight">Cortex</span>
            <span className="font-light text-lg opacity-60">UI</span>
          </div>
        )}
      </div>

      <nav className="flex-1 p-4 space-y-2" aria-label="Primary navigation">
        {navItems.map((item) => (
          <NavItem key={item.to} {...item} isCollapsed={isCollapsed} />
        ))}
      </nav>

      <div className="p-4 border-t border-white/5">
        <button
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="w-full flex items-center gap-3 px-4 py-3 rounded-xl text-white/50 hover:text-white hover:bg-white/5 transition-all"
          aria-label={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {isCollapsed ? <ChevronsRight className="w-5 h-5" /> : <ChevronsLeft className="w-5 h-5" />}
          {!isCollapsed && <span className="font-medium text-sm">Collapse</span>}
        </button>
      </div>

      <div className="p-4 border-t border-white/5">
        <button
          onClick={logout}
          className="w-full flex items-center gap-3 px-4 py-3 rounded-xl text-white/50 hover:text-white hover:bg-white/5 transition-all"
        >
          <LogOut className="w-5 h-5" />
          {!isCollapsed && <span className="font-medium text-sm">Logout</span>}
        </button>
      </div>

      <div className="p-4 border-t border-white/5">
        {!isCollapsed && (
          <div className="flex items-center justify-between text-xs text-white/30">
            <span>EmailOps Cortex</span>
            <span className="font-mono">v3.1.0</span>
          </div>
        )}
      </div>
    </aside>
  );
};
