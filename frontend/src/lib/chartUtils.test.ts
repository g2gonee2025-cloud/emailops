import { describe, it, expect } from 'vitest';
import {
    mapMetricsToChartData,
    xAxisTickFormatter,
    yAxisTickFormatter,
    tooltipValueFormatter,
    tooltipLabelFormatter,
} from './chartUtils';

describe('chartUtils', () => {
    describe('mapMetricsToChartData', () => {
        it('should return an empty array if metrics are undefined', () => {
            expect(mapMetricsToChartData(undefined)).toEqual([]);
        });

        it('should map metrics to chart data, preserving the full date string', () => {
            const metrics = {
                '2023-01-01': 100,
                '2023-01-02': 150,
            };
            const expected = [
                { name: '2023-01-01', value: 100 },
                { name: '2023-01-02', value: 150 },
            ];
            expect(mapMetricsToChartData(metrics)).toEqual(expected);
        });

        it('should handle an empty metrics object', () => {
            expect(mapMetricsToChartData({})).toEqual([]);
        });
    });

    describe('xAxisTickFormatter', () => {
        it('should format a date string to M/D format', () => {
            expect(xAxisTickFormatter('2023-01-01')).toBe('1/1');
            expect(xAxisTickFormatter('2023-12-25')).toBe('12/25');
        });
    });

    describe('yAxisTickFormatter', () => {
        it('should format numbers using compact notation for large numbers', () => {
            expect(yAxisTickFormatter(1000000)).toBe('1M');
        });

        it('should format numbers using standard notation for small numbers', () => {
            expect(yAxisTickFormatter(1234)).toBe('1,234');
        });
    });

    describe('tooltipValueFormatter', () => {
        it('should format numbers', () => {
            expect(tooltipValueFormatter(5678)).toBe('5,678');
        });

        it('should return strings as is', () => {
            expect(tooltipValueFormatter('test')).toBe('test');
        });
    });

    describe('tooltipLabelFormatter', () => {
        it('should format a date string', () => {
            expect(tooltipLabelFormatter('2023-01-01')).toBe('Jan 1, 2023');
        });

        it('should handle different date formats', () => {
            expect(tooltipLabelFormatter(new Date('2024-03-15').toString())).toBe('Mar 15, 2024');
        });
    });
});
