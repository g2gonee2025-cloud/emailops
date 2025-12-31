import { useState } from 'react';
import { Check, Copy, Download, AlertTriangle, XCircle, Info } from 'lucide-react';
import GlassCard from '../ui/GlassCard';
import { Button } from '../ui/Button';
import { StatusIndicator } from '../ui/StatusIndicator';
import type { DoctorReport } from '../../schemas/admin';
import { cn } from '../../lib/utils'; // Assuming utils exists

interface DoctorPanelProps {
  report: DoctorReport | null;
  isLoading?: boolean;
  onRefresh?: () => void;
  className?: string;
}

export default function DoctorPanel({
  report,
  isLoading,
  onRefresh,
  className,
}: DoctorPanelProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    if (!report) return;
    navigator.clipboard.writeText(JSON.stringify(report, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    if (!report) return;
    const blob = new Blob([JSON.stringify(report, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `doctor-report-${new Date().toISOString()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  if (!report && !isLoading) {
    return (
      <GlassCard className={cn('p-6 text-center text-white/50', className)}>
        No report data available.
      </GlassCard>
    );
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pass':
      case 'healthy':
        return <Check className="w-5 h-5 text-emerald-400" />;
      case 'fail':
      case 'unhealthy':
        return <XCircle className="w-5 h-5 text-red-400" />;
      case 'warn':
      case 'degraded':
        return <AlertTriangle className="w-5 h-5 text-yellow-400" />;
      default:
        return <Info className="w-5 h-5 text-blue-400" />;
    }
  };

  return (
    <GlassCard className={cn('p-6', className)} data-testid="doctor-panel">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-xl font-bold flex items-center gap-3">
          System Health Doctor
          {report && (
            <StatusIndicator
              status={
                report.overall_status === 'healthy'
                  ? 'healthy'
                  : report.overall_status === 'unhealthy'
                  ? 'critical'
                  : 'warning'
              }
            />
          )}
        </h2>
        <div className="flex gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleCopy}
            disabled={!report || isLoading}
            title="Copy Report"
          >
            {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleDownload}
            disabled={!report || isLoading}
            title="Download JSON"
          >
            <Download className="w-4 h-4" />
          </Button>
          {onRefresh && (
            <Button
              variant="outline"
              size="sm"
              onClick={onRefresh}
              disabled={isLoading}
            >
              {isLoading ? 'Running...' : 'Run Doctor'}
            </Button>
          )}
        </div>
      </div>

      <div className="space-y-4">
        {report?.checks.map((check, index) => (
          <div
            key={index}
            className="flex items-start gap-4 p-4 rounded-lg bg-white/5 border border-white/10"
          >
            <div className="mt-1">{getStatusIcon(check.status)}</div>
            <div className="flex-1">
              <div className="flex justify-between items-start">
                <h3 className="font-semibold text-white/90">{check.name}</h3>
                <span className="text-xs font-mono uppercase opacity-50 bg-white/10 px-2 py-1 rounded">
                  {check.status}
                </span>
              </div>
              {check.message && (
                <p className="mt-1 text-sm text-white/60">{check.message}</p>
              )}
              {check.details && Object.keys(check.details).length > 0 && (
                <pre className="mt-3 text-xs bg-black/30 p-3 rounded overflow-x-auto text-white/50">
                  {JSON.stringify(check.details, null, 2)}
                </pre>
              )}
            </div>
          </div>
        ))}
      </div>
    </GlassCard>
  );
}
