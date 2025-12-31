import { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { cn } from '../lib/utils';
import { useAuth } from '../contexts/AuthContext';
import {
  LayoutDashboard,
  MessageSquare,
  Search,
  Upload,
  Settings,
  ShieldCheck,
  PenTool,
  LogOut,
  ChevronLeft,
  type LucideIcon,
} from 'lucide-react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/Tooltip';
import { Button } from './ui/Button';

// Navigation Item Component
interface NavItemProps {
  icon: LucideIcon;
  label: string;
  to: string;
  isCollapsed: boolean;
}

const NavItem = ({ icon: Icon, label, to, isCollapsed }: NavItemProps) => {
  const location = useLocation();
  const isActive = location.pathname === to || (to === '/search' && location.pathname.startsWith('/thread'));

  const linkContent = (
    <>
      <Icon
        className={cn(
          'w-5 h-5 transition-colors',
          isActive ? "text-emerald-300" : "text-white/70 group-hover:text-white"
        )}
      />
      {!isCollapsed && <span className="font-medium text-sm">{label}</span>}
      {!isCollapsed && isActive && (
        <div className="ml-auto w-1.5 h-1.5 rounded-full bg-emerald-300 animate-pulse" />
      )}
    </>
  );

  const linkClasses = cn(
    "w-full flex items-center gap-3 rounded-xl transition-all duration-200 group",
    isCollapsed ? "px-3.5 py-3" : "px-4 py-3",
    isActive
      ? "bg-emerald-500/15 text-emerald-200 border border-emerald-400/30"
      : "text-white/50 hover:text-white hover:bg-white/5 border border-transparent"
  );

  if (isCollapsed) {
    return (
      <TooltipProvider>
        <Tooltip delayDuration={100}>
          <TooltipTrigger asChild>
            <NavLink to={to} className={linkClasses} aria-label={label}>
              {linkContent}
            </NavLink>
          </TooltipTrigger>
          <TooltipContent side="right">
            <p>{label}</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  }

  return (
    <NavLink to={to} className={linkClasses} title={label}>
      {linkContent}
    </NavLink>
  );
};

export const Sidebar = () => {
  const { logout } = useAuth();
  const [isCollapsed, setIsCollapsed] = useState(false);

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
    <aside className={cn("flex flex-col z-20 bg-black/40 backdrop-blur-xl border-r border-white/5 transition-all duration-300 ease-in-out", isCollapsed ? "w-20" : "w-64")} role="navigation" aria-label="Main navigation">
      {/* Logo and Collapse Toggle */}
      <div className="h-16 flex items-center border-b border-white/5 gap-3 relative">
        {!isCollapsed && (
          <div className={cn("flex items-center gap-3 pl-6", { "opacity-0": isCollapsed })}>
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-emerald-400 to-cyan-500 flex items-center justify-center shadow-lg shadow-emerald-500/30" aria-hidden="true">
              <ShieldCheck className="text-white" size={18} />
            </div>
            <div>
              <span className="font-display font-semibold text-lg tracking-tight">Cortex</span>
              <span className="font-light text-lg opacity-60">UI</span>
            </div>
          </div>
        )}
        <Button
          onClick={() => setIsCollapsed(!isCollapsed)}
          variant="ghost"
          size="icon"
          className="absolute top-1/2 -translate-y-1/2 -right-3.5 z-30 h-7 w-7 rounded-full border border-white/10 bg-white/5 text-white/60 hover:bg-white/10 hover:text-white"
          aria-label={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          <ChevronLeft className={cn("w-4 h-4 transition-transform", { "rotate-180": isCollapsed })} />
        </Button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-2" aria-label="Primary navigation">
        {navItems.map((item) => (
          <NavItem
            key={item.to}
            {...item}
            isCollapsed={isCollapsed}
          />
        ))}
      </nav>

      {/* Logout Button */}
      <div className="p-4 border-t border-white/5">
        <Button
          onClick={logout}
          variant="ghost"
          className={cn(
            "w-full gap-3 rounded-xl text-white/50 hover:text-white hover:bg-white/5",
            isCollapsed ? "px-3.5 py-3 justify-center" : "px-4 py-3 justify-start"
          )}
          aria-label={isCollapsed ? "Logout" : undefined}
        >
          <LogOut className="w-5 h-5" />
          {!isCollapsed && <span className="font-medium text-sm">Logout</span>}
        </Button>
      </div>

      {/* Footer */}
      <div className={cn("p-4 border-t border-white/5 transition-opacity", { "opacity-0": isCollapsed })}>
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
