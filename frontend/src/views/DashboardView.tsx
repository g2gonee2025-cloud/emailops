import { useState, useEffect, useRef } from 'react';
import GlassCard from './ui/GlassCard';
import { StatusIndicator } from './ui/StatusIndicator';
import { api } from '../lib/api';
import type { HealthResponse } from '../lib/api';
import { cn } from '../lib/utils';
import {
  Activity,
  Database,
  Mail,
  Shield,
  TrendingUp,
  Zap,
  Server,
  Clock
} from 'lucide-react';

interface LogEntry {
  logId: number;
  timestamp: string;
  message: string;
  level: 'INFO' | 'WARN' | 'ERROR' | 'DEBUG';
}

export function DashboardView() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const logIdCounter = useRef(0);

  useEffect(() => {
    // Fetch health
    api.fetchHealth().then(setHealth);

    // Simulate log stream
    const messages = [
      "Pipeline check complete",
      "Index optimization running",
      "Embedding batch processed",
      "Health check passed",
      "Queue depth: 12 items",
      "Connection pool: 8/10 active",
    ];

    const interval = setInterval(() => {
      const levels: LogEntry['level'][] = ['INFO', 'DEBUG', 'INFO', 'WARN'];
      const newLog: LogEntry = {
        logId: logIdCounter.current++,
        timestamp: new Date().toLocaleTimeString(),
        message: messages[Math.floor(Math.random() * messages.length)],
        level: levels[Math.floor(Math.random() * levels.length)],
      };
      setLogs(prev => [...prev.slice(-19), newLog]);
    }, 2500);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="p-8 space-y-8 animate-fade-in">
      {/* Header */}
      <header className="flex justify-between items-center">
        <div>
          <h1 className="text-4xl font-bold tracking-tight mb-2 bg-gradient-to-r from-white to-white/60 bg-clip-text text-transparent">
            Mission Control
          </h1>
          <p className="text-white/40">Real-time system monitoring and ingestion status.</p>
        </div>
        <div className="flex gap-4">
          <GlassCard className="px-4 py-2 flex items-center gap-3">
            <StatusIndicator status={health?.status === 'healthy' ? 'healthy' : 'warning'} />
            <span className="text-sm font-medium">{health?.status?.toUpperCase() || 'CHECKING...'}</span>
            <div className="h-4 w-px bg-white/10" />
            <span className="text-xs font-mono text-white/50">v{health?.version || '...'}</span>
          </GlassCard>
        </div>
      </header>

      {/* KPI Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <KPICard
          title="Pipeline Throughput"
          value="1,240"
          unit="/hr"
          icon={Zap}
          trend="+12%"
          trendUp
          color="emerald"
        />
        <KPICard
          title="Active Connections"
          value="8"
          unit="nodes"
          icon={Server}
          trend="stable"
          color="blue"
        />
        <KPICard
          title="Vector Index"
          value="84.2k"
          unit="chunks"
          icon={Database}
          trend="+540"
          trendUp
          color="purple"
        />
        <KPICard
          title="Security Gate"
          value="99.9"
          unit="%"
          icon={Shield}
          color="green"
        />
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Live Stream */}
        <div className="lg:col-span-2">
          <GlassCard className="h-[450px] flex flex-col">
            <div className="p-4 border-b border-white/5 flex justify-between items-center">
              <h3 className="font-semibold flex items-center gap-2">
                <Activity className="w-4 h-4 text-blue-400" />
                Live Process Stream
              </h3>
              <StatusIndicator status="healthy" pulsing />
            </div>
            <div className="flex-1 overflow-y-auto p-4 font-mono text-sm space-y-2">
              {logs.length === 0 && (
                <div className="h-full flex items-center justify-center text-white/20">
                  Waiting for events...
                </div>
              )}
              {logs.map((log) => (
                <div key={log.logId} className="flex gap-4 animate-slide-up">
                  <span className="text-white/30 text-xs w-20">{log.timestamp}</span>
                  <span className={cn(
                    "font-bold text-xs uppercase w-12",
                    log.level === 'ERROR' ? "text-red-400" :
                    log.level === 'WARN' ? "text-yellow-400" :
                    log.level === 'DEBUG' ? "text-purple-400" :
                    "text-blue-400"
                  )}>{log.level}</span>
                  <span className="text-white/70">{log.message}</span>
                </div>
              ))}
            </div>
          </GlassCard>
        </div>

        {/* Quick Stats */}
        <div className="space-y-6">
          <GlassCard className="p-6">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <Clock className="w-4 h-4 text-orange-400" />
              System Uptime
            </h3>
            <div className="text-3xl font-bold mb-2">99.98%</div>
            <p className="text-sm text-white/40">Last 30 days</p>
            <div className="mt-4 h-2 bg-white/5 rounded-full overflow-hidden">
              <div className="h-full w-[99.98%] bg-gradient-to-r from-green-500 to-emerald-400 rounded-full" />
            </div>
          </GlassCard>

          <GlassCard className="p-6">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <Mail className="w-4 h-4 text-blue-400" />
              Email Processing
            </h3>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-white/60">Today</span>
                <span className="font-bold">2,847</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-white/60">This Week</span>
                <span className="font-bold">18,492</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-white/60">Pending</span>
                <span className="font-bold text-yellow-400">48</span>
              </div>
            </div>
          </GlassCard>

          <GlassCard className="p-6">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-green-400" />
              RAG Performance
            </h3>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-white/60">Avg Latency</span>
                  <span className="font-mono">142ms</span>
                </div>
                <div className="h-2 bg-white/5 rounded-full overflow-hidden">
                  <div className="h-full w-[35%] bg-green-500 rounded-full" />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-white/60">Cache Hit Rate</span>
                  <span className="font-mono">87%</span>
                </div>
                <div className="h-2 bg-white/5 rounded-full overflow-hidden">
                  <div className="h-full w-[87%] bg-blue-500 rounded-full" />
                </div>
              </div>
            </div>
          </GlassCard>
        </div>
      </div>
    </div>
  );
}

// KPI Card Component
interface KPICardProps {
  title: string;
  value: string;
  unit?: string;
  icon: React.ElementType;
  trend?: string;
  trendUp?: boolean;
  color?: 'emerald' | 'blue' | 'purple' | 'green' | 'orange';
}

function KPICard({ title, value, unit, icon: Icon, trend, trendUp, color = 'blue' }: KPICardProps) {
  const colorClasses = {
    emerald: 'bg-emerald-500/10 text-emerald-400',
    blue: 'bg-blue-500/10 text-blue-400',
    purple: 'bg-purple-500/10 text-purple-400',
    green: 'bg-green-500/10 text-green-400',
    orange: 'bg-orange-500/10 text-orange-400',
  };

  return (
    <GlassCard className="p-6 relative group">
      <div className={cn("absolute top-4 right-4 p-2 rounded-lg transition-colors", colorClasses[color])}>
        <Icon className="w-5 h-5" />
      </div>
      <h3 className="text-sm font-medium text-white/50">{title}</h3>
      <div className="mt-2 text-3xl font-bold">
        {value}
        {unit && <span className="text-sm font-normal opacity-50 ml-1">{unit}</span>}
      </div>
      {trend && (
        <div className={cn(
          "mt-3 text-xs font-medium",
          trendUp ? "text-green-400" : "text-white/40"
        )}>
          {trend}
        </div>
      )}
    </GlassCard>
  );
}
