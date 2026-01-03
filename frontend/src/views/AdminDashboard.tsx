
import { useEffect, useMemo } from 'react';
import GlassCard from '../components/ui/GlassCard';
import { StatusIndicator } from '../components/ui/StatusIndicator';
import { Activity, Server, Settings, Play, CheckCircle, XCircle, AlertTriangle } from 'lucide-react';
import { cn } from '../lib/utils';
import { redactObject } from '../schemas/admin';
import type { AppConfig, DoctorCheck } from '../schemas/admin';
import { useAdmin } from '../hooks/useAdmin';
import { Skeleton } from '../components/ui/Skeleton';
import { Alert, AlertDescription, AlertTitle } from '../components/ui/Alert';
import { useToast } from '../contexts/toastContext';
import { Button } from '../components/ui/Button';
import ConfigPanel from '../components/admin/ConfigPanel';

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

export default function AdminDashboard() {
    const { addToast } = useToast();
    const {
        status, isStatusLoading, statusError,
        config, isConfigLoading, configError,
        runDoctor, isDoctorRunning, doctorReport, doctorError,
    } = useAdmin();

    useEffect(() => {
        if (statusError) {
            addToast({
                type: 'error',
                message: "Error loading status",
                details: statusError.message,
            });
        }
    }, [statusError, addToast]);

    useEffect(() => {
        if (configError) {
            addToast({
                type: 'error',
                message: "Error loading configuration",
                details: configError.message,
            });
        }
    }, [configError, addToast]);

    useEffect(() => {
        if (doctorError) {
            addToast({
                type: 'error',
                message: "Error running diagnostics",
                details: doctorError.message,
            });
        }
    }, [doctorError, addToast]);

    const handleSaveConfig = (_newConfig: AppConfig) => {
        addToast({
            type: 'warning',
            message: "Configuration Save Not Available",
            details: "Configuration updates require a backend API endpoint. Contact your administrator.",
        });
    };

    const redactedDoctorReport = useMemo(() => {
        if (!doctorReport) return null;
        const redactedChecks = doctorReport.checks.map((check: DoctorCheck) => ({
            ...check,
            details: redactObject(check.details) ?? undefined,
        }));
        return { ...doctorReport, checks: redactedChecks };
    }, [doctorReport]);


  return (
    <div className="p-8 space-y-8 pb-24">
      <header className="flex justify-between items-center">
        <div>
          <h1 className="text-4xl font-display font-semibold tracking-tight mb-2 bg-gradient-to-r from-white to-white/70 bg-clip-text text-transparent">
            System Administration
          </h1>
          <p className="text-white/40">Diagnostics and Configuration Control</p>
        </div>
        <div className="flex gap-4">
            <GlassCard className="px-4 py-2 flex items-center gap-2">
                <StatusIndicator status={status ? 'healthy' : 'inactive'} />
                {isStatusLoading ? <Skeleton className="h-5 w-16" /> : <span className="text-sm font-medium">{status?.env?.toUpperCase() || 'UNKNOWN'}</span>}
            </GlassCard>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Doctor Section */}
        <GlassCard className="p-6 space-y-6">
            <div className="flex justify-between items-center">
                <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-emerald-500/20 text-emerald-300">
                        <Activity className="w-6 h-6" />
                    </div>
                    <h2 className="text-xl font-semibold">System Doctor</h2>
                </div>
                <Button
                    onClick={() => runDoctor()}
                    disabled={isDoctorRunning}
                >
                    <Play className={cn("w-4 h-4 mr-2", isDoctorRunning && "animate-spin")} />
                    {isDoctorRunning ? 'Running...' : 'Run Diagnostics'}
                </Button>
            </div>

            <div className="space-y-4 min-h-[200px]">
                {doctorError && (
                    <Alert variant="destructive">
                        <AlertTriangle className="h-4 w-4" />
                        <AlertTitle>Diagnostics Failed</AlertTitle>
                        <AlertDescription>
                            An error occurred while running diagnostics. Please check the console for details.
                        </AlertDescription>
                    </Alert>
                )}

                {!redactedDoctorReport && !isDoctorRunning && !doctorError &&(
                    <div className="h-full flex flex-col items-center justify-center text-white/20 py-12">
                        <Activity className="w-12 h-12 mb-4 opacity-50" />
                        <p>Ready to scan system health</p>
                    </div>
                )}

                {redactedDoctorReport && (
                    <div className="space-y-3 animate-slide-up">
                         <div className={cn(
                             "p-3 rounded-lg border flex items-center justify-between mb-4",
                             redactedDoctorReport.overall_status === 'healthy' ? "bg-green-500/10 border-green-500/20" : "bg-red-500/10 border-red-500/20"
                         )}>
                             <span className="font-semibold">Overall Status</span>
                             <span className={cn(
                                 "uppercase font-bold tracking-wider text-sm",
                                 redactedDoctorReport.overall_status === 'healthy' ? "text-green-400" : "text-red-400"
                             )}>{redactedDoctorReport.overall_status}</span>
                         </div>

                        {redactedDoctorReport.checks.map((check: DoctorCheck, idx: number) => (
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
                    <div className="p-2 rounded-lg bg-emerald-500/20 text-emerald-300">
                        <Server className="w-6 h-6" />
                    </div>
                    <h2 className="text-xl font-semibold">Environment</h2>
                </div>
                {statusError && (
                    <Alert variant="destructive">
                        <AlertTriangle className="h-4 w-4" />
                        <AlertTitle>Error</AlertTitle>
                        <AlertDescription>
                            {statusError.message}
                        </AlertDescription>
                    </Alert>
                )}
                {isStatusLoading && (
                    <div className="grid grid-cols-2 gap-4">
                        <Skeleton className="h-16" />
                        <Skeleton className="h-16" />
                        <Skeleton className="h-16 col-span-2" />
                    </div>
                )}
                {status && (
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
                {configError && (
                    <Alert variant="destructive">
                        <AlertTriangle className="h-4 w-4" />
                        <AlertTitle>Error</AlertTitle>
                        <AlertDescription>
                            {configError.message}
                        </AlertDescription>
                    </Alert>
                )}
                {isConfigLoading && (
                    <div className="space-y-4">
                        <Skeleton className="h-8 w-full" />
                        <Skeleton className="h-8 w-full" />
                        <Skeleton className="h-8 w-full" />
                        <Skeleton className="h-8 w-full" />
                    </div>
                )}
                {config && (
                    <ConfigPanel
                        config={config as unknown as AppConfig}
                        onSave={handleSaveConfig}
                        isLoading={isConfigLoading}
                    />
                )}
            </GlassCard>
        </div>
      </div>
    </div>
  );
}
