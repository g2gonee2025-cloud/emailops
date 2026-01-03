import { Alert, AlertDescription, AlertTitle } from '../ui/Alert';
import { Skeleton } from '../ui/Skeleton';
import GlassCard from '../ui/GlassCard';
import { AlertTriangle, BarChart3 } from 'lucide-react';

export interface KPIData {
  readonly id: string;
  readonly title: string;
  readonly value: string;
  readonly change: string;
}

interface KPIGridProps {
  readonly kpis?: readonly KPIData[];
  readonly isLoading?: boolean;
  readonly error?: Error | null;
}

export default function KPIGrid({ kpis = [], isLoading = false, error = null }: KPIGridProps) {
  const renderContent = () => {
    if (isLoading) {
      return <KPISkeleton />;
    }
    if (error) {
      return <KPIError />;
    }
    if (kpis.length === 0) {
      return <KPIEmpty />;
    }
    return kpis.map((kpi) => <KPICard key={kpi.id} {...kpi} />);
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {renderContent()}
    </div>
  );
}

interface KPICardProps {
  readonly id: string;
  readonly title: string;
  readonly value: string;
  readonly change: string;
}

function KPICard({ id, title, value, change }: KPICardProps) {
  const isPositive = change.startsWith('+');
  return (
    <GlassCard className="p-5" data-testid={`kpi-card-${id}`}>
      <h3 className="text-sm font-medium text-white/50 uppercase tracking-wider">
        {title}
      </h3>
      <div className="mt-2 flex items-baseline">
        <p className="text-2xl font-semibold text-white">{value}</p>
        <span
          className={`ml-2 text-sm font-semibold ${
            isPositive ? 'text-green-400' : 'text-red-400'
          }`}
        >
          {change}
        </span>
      </div>
    </GlassCard>
  );
}

function KPISkeleton() {
  const skeletonItems = Array.from({ length: 4 }, (_, i) => `skeleton-${i}`);
  return (
    <>
      {skeletonItems.map((key) => (
        <GlassCard key={key} className="p-5">
          <Skeleton className="h-4 w-3/4" />
          <Skeleton className="h-8 w-1/2 mt-2" />
        </GlassCard>
      ))}
    </>
  );
}

function KPIError() {
  return (
    <div className="lg:col-span-4">
        <Alert variant="destructive">
        <AlertTriangle className="h-4 w-4" />
        <AlertTitle>Error Loading KPIs</AlertTitle>
        <AlertDescription>
            There was an error loading the KPI data. Please try again later.
        </AlertDescription>
        </Alert>
    </div>
  );
}

function KPIEmpty() {
  return (
    <div className="lg:col-span-4">
      <GlassCard className="p-8 flex flex-col items-center justify-center text-center">
        <BarChart3 className="w-12 h-12 text-white/20 mb-4" />
        <h3 className="text-lg font-medium text-white/60 mb-2">No KPI Data Available</h3>
        <p className="text-sm text-white/40">
          KPI metrics will appear here once data is available.
        </p>
      </GlassCard>
    </div>
  );
}
