import React from 'react';
import { cn } from '../../lib/utils';

interface StatusIndicatorProps {
  status: 'healthy' | 'warning' | 'critical' | 'inactive';
  label?: string;
  pulsing?: boolean;
}

export const StatusIndicator: React.FC<StatusIndicatorProps> = ({
  status,
  label,
  pulsing = true
}) => {
  const colors = {
    healthy: "bg-emerald-500",
    warning: "bg-amber-500",
    critical: "bg-red-500",
    inactive: "bg-slate-500",
  };

  const shadowColors = {
    healthy: "shadow-emerald-500/50",
    warning: "shadow-amber-500/50",
    critical: "shadow-red-500/50",
    inactive: "shadow-slate-500/50",
  };

  return (
    <div className="flex items-center space-x-2">
      <div className="relative flex items-center justify-center w-3 h-3">
        {pulsing && status !== 'inactive' && (
          <span className={cn(
            "absolute w-full h-full rounded-full opacity-75 animate-ping",
            colors[status]
          )} />
        )}
        <span className={cn(
          "relative w-2 h-2 rounded-full shadow-[0_0_8px]",
          colors[status],
          shadowColors[status]
        )} />
      </div>
      {label && (
        <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
          {label}
        </span>
      )}
    </div>
  );
};
