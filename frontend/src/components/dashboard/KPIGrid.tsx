import { Alert, AlertDescription, AlertTitle } from '../ui/Alert';
import { Skeleton } from '../ui/Skeleton';
import GlassCard from '../ui/GlassCard';
import { AlertTriangle } from 'lucide-react';

// Mock data for KPIs
const mockKpis = [
  { id: 'total-emails', title: 'Total Emails Processed', value: '1,234,567', change: '+5.2%' },
  { id: 'avg-response-time', title: 'Average Response Time', value: '2.3 hours', change: '-1.5%' },
  { id: 'open-rate', title: 'Open Rate', value: '25.8%', change: '+0.8%' },
  { id: 'click-through-rate', title: 'Click-Through Rate', value: '4.2%', change: '+0.2%' },
];

export default function KPIGrid() {
  const isLoading = false;
  const error = null;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {isLoading ? (
        <KPISkeleton />
      ) : error ? (
        <KPIError />
      ) : (
        mockKpis.map((kpi) => <KPICard key={kpi.id} {...kpi} />)
      )}
    </div>
  );
}

interface KPICardProps {
  id: string;
  title: string;
  value: string;
  change: string;
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
  return (
    <>
      {Array.from({ length: 4 }).map((_, index) => (
        <GlassCard key={index} className="p-5">
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
