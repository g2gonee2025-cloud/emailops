import { useState, useEffect } from 'react';
import GlassCard from './ui/GlassCard';
import { StatusIndicator } from './ui/StatusIndicator';
import { api } from '../lib/api';
import { Activity, Server, Settings, Play, CheckCircle, XCircle, AlertTriangle } from 'lucide-react';
import { cn } from '../lib/utils';

interface DoctorCheck {
  name: string;
  status: 'pass' | 'fail' | 'warn';
  message?: string;
  details?: Record<string, unknown>;
}

interface DoctorReport {
  overall_status: 'healthy' | 'degraded' | 'unhealthy';
  checks: DoctorCheck[];
}

interface StatusData {
    env: string;
    service: string;
    status: string;
}

export function AdminDashboard() {
  const [doctorReport, setDoctorReport] = useState<DoctorReport | null>(null);
  const [isRunningDoctor, setIsRunningDoctor] = useState(false);
  const [status, setStatus] = useState<StatusData | null>(null);
  const [config, setConfig] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    const [statusData, configData] = await Promise.all([
      api.fetchStatus(),
      api.fetchConfig()
    ]);
    setStatus(statusData as StatusData);
    setConfig(configData as unknown as Record<string, unknown>);
  };

  const runDiagnostics = async () => {
    setIsRunningDoctor(true);
    try {
        const report = await api.runDoctor();
        setDoctorReport(report as DoctorReport);
    } finally {
        setIsRunningDoctor(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pass': return 'text-green-400';
      case 'fail': return 'text-red-400';
      case 'warn': return 'text-yellow-400';
      default: return 'text-gray-400';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pass': return <CheckCircle className="w-5 h-5 text-green-400" />;
      case 'fail': return <XCircle className="w-5 h-5 text-red-400" />;
      case 'warn': return <AlertTriangle className="w-5 h-5 text-yellow-400" />;
      default: return null;
    }
  };

  return (
    <div className="p-8 space-y-8 animate-fade-in pb-24">
      <header className="flex justify-between items-center">
        <div>
          <h1 className="text-4xl font-bold tracking-tight mb-2 bg-gradient-to-r from-white to-white/60 bg-clip-text text-transparent">
            System Administration
          </h1>
          <p className="text-white/40">Diagnostics and Configuration Control</p>
        </div>
        <div className="flex gap-4">
            <GlassCard className="px-4 py-2 flex items-center gap-2">
                <StatusIndicator status={status ? 'healthy' : 'inactive'} />
                <span className="text-sm font-medium">{status?.env?.toUpperCase() || 'UNKNOWN'}</span>
            </GlassCard>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Doctor Section */}
        <GlassCard className="p-6 space-y-6">
            <div className="flex justify-between items-center">
                <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-blue-500/20 text-blue-400">
                        <Activity className="w-6 h-6" />
                    </div>
                    <h2 className="text-xl font-semibold">System Doctor</h2>
                </div>
                <button
                    onClick={runDiagnostics}
                    disabled={isRunningDoctor}
                    className="flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-50 transition-all font-medium text-sm"
                >
                    <Play className={cn("w-4 h-4", isRunningDoctor && "animate-spin")} />
                    {isRunningDoctor ? 'Running Diagnostics...' : 'Run Diagnostics'}
                </button>
            </div>

            <div className="space-y-4 min-h-[200px]">
                {!doctorReport && !isRunningDoctor && (
                    <div className="h-full flex flex-col items-center justify-center text-white/20 py-12">
                        <Activity className="w-12 h-12 mb-4 opacity-50" />
                        <p>Ready to scan system health</p>
                    </div>
                )}

                {doctorReport && (
                    <div className="space-y-3 animate-slide-up">
                         <div className={cn(
                             "p-3 rounded-lg border flex items-center justify-between mb-4",
                             doctorReport.overall_status === 'healthy' ? "bg-green-500/10 border-green-500/20" : "bg-red-500/10 border-red-500/20"
                         )}>
                             <span className="font-semibold">Overall Status</span>
                             <span className={cn(
                                 "uppercase font-bold tracking-wider text-sm",
                                 doctorReport.overall_status === 'healthy' ? "text-green-400" : "text-red-400"
                             )}>{doctorReport.overall_status}</span>
                         </div>

                        {doctorReport.checks.map((check, idx) => (
                            <div key={idx} className="flex items-center justify-between p-3 rounded-lg bg-white/5 border border-white/5 hover:border-white/10 transition-colors">
                                <div className="flex items-center gap-3">
                                    {getStatusIcon(check.status)}
                                    <span>{check.name}</span>
                                </div>
                                <div className="text-right">
                                    <span className={cn("text-sm block", getStatusColor(check.status))}>
                                        {check.message || check.status.toUpperCase()}
                                    </span>
                                    {check.details && (
                                        <pre className="text-xs text-white/30 mt-1 max-w-[200px] overflow-hidden text-ellipsis">
                                            {JSON.stringify(check.details)}
                                        </pre>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </GlassCard>

        <div className="space-y-8">
            {/* Environment Status */}
            <GlassCard className="p-6">
                <div className="flex items-center gap-3 mb-6">
                    <div className="p-2 rounded-lg bg-purple-500/20 text-purple-400">
                        <Server className="w-6 h-6" />
                    </div>
                    <h2 className="text-xl font-semibold">Environment</h2>
                </div>
                {status ? (
                    <div className="grid grid-cols-2 gap-4">
                        <div className="p-3 rounded-lg bg-white/5 border border-white/5">
                            <span className="text-xs text-white/40 block mb-1">Service</span>
                            <span className="font-mono text-sm">{status.service}</span>
                        </div>
                        <div className="p-3 rounded-lg bg-white/5 border border-white/5">
                            <span className="text-xs text-white/40 block mb-1">Environment</span>
                            <span className="font-mono text-sm">{status.env}</span>
                        </div>
                        <div className="p-3 rounded-lg bg-white/5 border border-white/5 col-span-2">
                             <span className="text-xs text-white/40 block mb-1">System Status</span>
                             <span className="flex items-center gap-2 text-green-400 font-mono text-sm">
                                <Activity className="w-3 h-3" />
                                {status.status?.toUpperCase()}
                             </span>
                        </div>
                    </div>
                ) : (
                    <div className="text-white/20 text-center py-8">Loading status...</div>
                )}
            </GlassCard>

             {/* Config Viewer */}
             <GlassCard className="p-6 flex-1">
                <div className="flex items-center gap-3 mb-6">
                    <div className="p-2 rounded-lg bg-orange-500/20 text-orange-400">
                        <Settings className="w-6 h-6" />
                    </div>
                    <h2 className="text-xl font-semibold">Configuration</h2>
                </div>
                {config ? (
                    <div className="space-y-4">
                        {Object.entries(config).map(([key, value]) => (
                             <div key={key} className="flex justify-between items-center py-2 border-b border-white/5 last:border-0">
                                <span className="text-white/60 capitalize">{key.replace(/_/g, ' ')}</span>
                                <span className="font-mono text-sm px-2 py-0.5 rounded bg-white/10">
                                    {String(value)}
                                </span>
                             </div>
                        ))}
                    </div>
                ) : (
                    <div className="text-white/20 text-center py-8">Loading configuration...</div>
                )}
            </GlassCard>
        </div>
      </div>
    </div>
  );
}
