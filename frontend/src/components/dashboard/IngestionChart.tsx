import { useMemo } from 'react';
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  mapToChartData,
  xAxisFormatter,
  yAxisFormatter,
  tooltipFormatter,
} from '../../lib/chartUtils';
import GlassCard from '../ui/GlassCard';

interface IngestionMetric {
  date: string;
  throughput: number; // items per hour
  latency: number; // ms
}

interface IngestionChartProps {
  data: IngestionMetric[];
  className?: string;
  height?: number;
}

export default function IngestionChart({
  data,
  className,
  height = 300,
}: IngestionChartProps) {
  const chartData = useMemo(() => {
    return mapToChartData(data, 'date', 'throughput', 'throughput').map(
      (point, index) => ({
        ...point,
        latency: data[index]?.latency || 0,
      })
    );
  }, [data]);

  return (
    <GlassCard className={className} data-testid="ingestion-chart">
      <div className="p-6">
        <h3 className="text-lg font-semibold mb-6 flex items-center justify-between">
          <span>Ingestion Performance</span>
          <div className="flex gap-4 text-xs font-medium">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              <span className="text-white/60">Throughput/hr</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-amber-400" />
              <span className="text-white/60">Avg Latency (ms)</span>
            </div>
          </div>
        </h3>
        <div style={{ height }} className="w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="colorThroughput" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#34d399" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#34d399" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="colorLatency" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="rgba(255,255,255,0.05)"
                vertical={false}
              />
              <XAxis
                dataKey="date"
                stroke="rgba(255,255,255,0.3)"
                fontSize={12}
                tickLine={false}
                axisLine={false}
                tickFormatter={xAxisFormatter}
              />
              <YAxis
                yAxisId="left"
                stroke="rgba(255,255,255,0.3)"
                fontSize={12}
                tickLine={false}
                axisLine={false}
                tickFormatter={yAxisFormatter}
              />
              <YAxis
                yAxisId="right"
                orientation="right"
                stroke="rgba(255,255,255,0.3)"
                fontSize={12}
                tickLine={false}
                axisLine={false}
                tickFormatter={(val) => `${val}ms`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(23, 23, 23, 0.9)',
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: '8px',
                  backdropFilter: 'blur(8px)',
                }}
                itemStyle={{ color: '#fff', fontSize: '12px' }}
                labelStyle={{
                  color: 'rgba(255,255,255,0.5)',
                  fontSize: '11px',
                  marginBottom: '4px',
                }}
                formatter={tooltipFormatter}
                labelFormatter={xAxisFormatter}
              />
              <Area
                yAxisId="left"
                type="monotone"
                dataKey="throughput"
                name="Throughput"
                stroke="#34d399"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#colorThroughput)"
              />
              <Area
                yAxisId="right"
                type="monotone"
                dataKey="latency"
                name="Latency"
                stroke="#f59e0b"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#colorLatency)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </GlassCard>
  );
}
