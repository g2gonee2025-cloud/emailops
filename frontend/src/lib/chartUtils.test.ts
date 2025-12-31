import { describe, it, expect } from 'vitest';
import {
  mapToChartData,
  mergeChartData,
  xAxisFormatter,
  yAxisFormatter,
  tooltipFormatter,
  type TimeSeriesDataPoint,
} from './chartUtils';

describe('Chart Utils', () => {
  // ===========================================================================
  // mapToChartData
  // ===========================================================================
  describe('mapToChartData', () => {
    it('should correctly map raw data to chart data format', () => {
      const rawData = [
        { day: '2023-01-01T00:00:00Z', count: 100 },
        { day: '2023-01-02T00:00:00Z', count: 150 },
      ];
      const expected: TimeSeriesDataPoint[] = [
        { date: '2023-01-01T00:00:00Z', 'MyMetric': 100 },
        { date: '2023-01-02T00:00:00Z', 'MyMetric': 150 },
      ];
      const result = mapToChartData(rawData, 'day', 'count', 'MyMetric');
      expect(result).toEqual(expected);
    });

    it('should handle empty input array', () => {
      const result = mapToChartData([], 'day', 'count', 'MyMetric');
      expect(result).toEqual([]);
    });

    it('should handle null or undefined input', () => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const result = mapToChartData(null as any, 'day', 'count', 'MyMetric');
        expect(result).toEqual([]);
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const result2 = mapToChartData(undefined as any, 'day', 'count', 'MyMetric');
        expect(result2).toEqual([]);
    });
  });

  // ===========================================================================
  // mergeChartData
  // ===========================================================================
  describe('mergeChartData', () => {
    it('should merge two datasets with overlapping dates', () => {
      const data1: TimeSeriesDataPoint[] = [
        { date: '2023-01-01T00:00:00Z', clicks: 10 },
        { date: '2023-01-02T00:00:00Z', clicks: 15 },
      ];
      const data2: TimeSeriesDataPoint[] = [
        { date: '2023-01-01T00:00:00Z', views: 100 },
        { date: '2023-01-03T00:00:00Z', views: 120 },
      ];
      const expected: TimeSeriesDataPoint[] = [
        { date: '2023-01-01T00:00:00Z', clicks: 10, views: 100 },
        { date: '2023-01-02T00:00:00Z', clicks: 15 },
        { date: '2023-01-03T00:00:00Z', views: 120 },
      ];
      const result = mergeChartData(data1, data2);
      expect(result).toEqual(expected);
    });

    it('should handle non-overlapping datasets', () => {
        const data1: TimeSeriesDataPoint[] = [
          { date: '2023-01-01T00:00:00Z', clicks: 10 },
        ];
        const data2: TimeSeriesDataPoint[] = [
          { date: '2023-01-02T00:00:00Z', views: 100 },
        ];
        const expected: TimeSeriesDataPoint[] = [
          { date: '2023-01-01T00:00:00Z', clicks: 10 },
          { date: '2023-01-02T00:00:00Z', views: 100 },
        ];
        const result = mergeChartData(data1, data2);
        expect(result).toEqual(expected);
    });

    it('should sort the merged data by date', () => {
      const data1: TimeSeriesDataPoint[] = [
        { date: '2023-01-03T00:00:00Z', value: 3 },
      ];
      const data2: TimeSeriesDataPoint[] = [
        { date: '2023-01-01T00:00:00Z', value: 1 },
      ];
      const expected: TimeSeriesDataPoint[] = [
        { date: '2023-01-01T00:00:00Z', value: 1 },
        { date: '2023-01-03T00:00:00Z', value: 3 },
      ];
      const result = mergeChartData(data1, data2);
      expect(result.map(d => d.date)).toEqual(expected.map(d => d.date));
    });

    it('should handle empty arrays', () => {
        const data1: TimeSeriesDataPoint[] = [
            { date: '2023-01-01T00:00:00Z', clicks: 10 },
          ];
        const result = mergeChartData([], data1, []);
        expect(result).toEqual(data1);
    });
  });

  // ===========================================================================
  // Formatters
  // ===========================================================================
  describe('Formatters', () => {
    it('xAxisFormatter should format ISO date string', () => {
      expect(xAxisFormatter('2023-10-26T12:00:00Z')).toBe('Oct 26');
    });

    it('xAxisFormatter should return original string on invalid date', () => {
        expect(xAxisFormatter('invalid-date')).toBe('invalid-date');
    });

    it('yAxisFormatter should format numbers into compact form', () => {
      expect(yAxisFormatter(500)).toBe('500');
      expect(yAxisFormatter(1500)).toBe('2K'); // toFixed(0) rounds
      expect(yAxisFormatter(12345)).toBe('12K');
      expect(yAxisFormatter(1_234_567)).toBe('1.2M');
    });

    it('tooltipFormatter should format numbers with commas', () => {
        expect(tooltipFormatter(1234567)).toBe('1,234,567');
        expect(tooltipFormatter('N/A')).toBe('N/A');
    });
  });
});
