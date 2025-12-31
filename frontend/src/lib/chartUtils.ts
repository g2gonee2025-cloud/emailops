import { format, parseISO } from 'date-fns';

/**
 * A generic data point for a time-series chart.
 * The `date` is a string in ISO 8601 format (e.g., '2023-01-01T00:00:00Z').
 * All other keys are dynamic and represent the metrics being plotted.
 */
export type TimeSeriesDataPoint = {
  date: string;
  [key: string]: number | string;
};

/**
 * Maps an array of metric objects into a format compatible with Recharts.
 *
 * @param data - The array of data points from the API.
 * @param dateKey - The key in each data point that represents the date.
 * @param valueKey - The key in each data point that represents the value to be plotted.
 * @param name - The name to assign to the plotted value in the chart data.
 * @returns A new array of `TimeSeriesDataPoint` objects.
 */
export function mapToChartData<T extends object>(
  data: T[],
  dateKey: keyof T,
  valueKey: keyof T,
  name: string,
): TimeSeriesDataPoint[] {
  if (!data) return [];
  return data.map((item) => ({
    date: item[dateKey] as string,
    [name]: item[valueKey] as number,
  }));
}

/**
 * Merges multiple Recharts datasets into a single dataset.
 * It uses the `date` field as the key for merging.
 *
 * @param datasets - An array of `TimeSeriesDataPoint` arrays to merge.
 * @returns A single merged array of `TimeSeriesDataPoint` objects.
 */
export function mergeChartData(
  ...datasets: TimeSeriesDataPoint[][]
): TimeSeriesDataPoint[] {
  const merged = new Map<string, TimeSeriesDataPoint>();

  datasets.forEach((dataset) => {
    dataset.forEach((item) => {
      const existing = merged.get(item.date) || { date: item.date };
      merged.set(item.date, { ...existing, ...item });
    });
  });

  return Array.from(merged.values()).sort(
    (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()
  );
}


/**
 * Formatter for the X-axis (date).
 * Shows month and day.
 * @param tick - The date string (ISO format).
 */
export const xAxisFormatter = (tick: string) => {
    try {
      return format(parseISO(tick), 'MMM d');
    } catch (_e) {
      return tick;
    }
  };


/**
 * Formatter for the Y-axis (value).
 * Displays large numbers in a compact format (e.g., 1000 -> 1K).
 * @param tick - The number to format.
 */
export const yAxisFormatter = (tick: number): string => {
    if (tick >= 1_000_000) {
      return `${(tick / 1_000_000).toFixed(1)}M`;
    }
    if (tick >= 1_000) {
      return `${(tick / 1_000).toFixed(0)}K`;
    }
    return tick.toString();
  };

/**
 * Formatter for the chart tooltip.
 * @param value - The value of the data point.
 */
export const tooltipFormatter = (value: number | string) => {
  if (typeof value === 'number') {
    return value.toLocaleString();
  }
  return value;
};
