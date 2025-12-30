import { formatNumber } from "./utils";

/**
 * Maps a generic metrics object to a Recharts-compatible data array.
 * Assumes the API returns an object where keys are date strings (e.g., "YYYY-MM-DD")
 * and values are numbers.
 *
 * @example
 * const metrics = { "2023-01-01": 100, "2023-01-02": 150 };
 * const rechartsData = mapMetricsToChartData(metrics);
 * // Output: [{ name: "1/1", value: 100 }, { name: "1/2", value: 150 }]
 */
export function mapMetricsToChartData(metrics: Record<string, number> | undefined) {
    if (!metrics) {
        return [];
    }
    // Note: Preserve the full date string as the 'name' for the tooltip formatter.
    // The visual formatting of the axis tick should be handled by a separate formatter.
    return Object.entries(metrics).map(([date, value]) => ({
        name: date,
        value,
    }));
}

/**
 * Formatter for the X-axis tick of a Recharts chart.
 * Takes a full date string (e.g., "2023-01-15") and returns a short format (e.g., "1/15").
 *
 * @param date The date string to format.
 * @returns A formatted string (e.g., "M/D").
 */
export function xAxisTickFormatter(date: string) {
    return new Date(date).toLocaleDateString('en-US', {
        month: 'numeric',
        day: 'numeric',
    });
}

/**
 * Formatter for the Y-axis tick of a Recharts chart.
 * Uses the compact number format for large numbers.
 *
 * @param value The number to format.
 * @returns A formatted string.
 */
export function yAxisTickFormatter(value: number) {
    return formatNumber(value);
}

/**
 * Formatter for the tooltip payload value.
 *
 * @param value The value from the tooltip payload.
 * @returns A formatted string.
 */
export function tooltipValueFormatter(value: number | string) {
    return typeof value === 'number' ? formatNumber(value) : value;
}

/**
 * Formatter for the tooltip label.
 *
 * @param label The label from the tooltip.
 * @returns A formatted date string.
 */
export function tooltipLabelFormatter(label: string) {
    // Note: This is a simplistic formatter. For more complex charts,
    // you might need to pass in the original date from the data payload.
    return new Date(label).toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric'
    });
}
