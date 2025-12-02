import { formatDistanceToNow } from 'date-fns';

/**
 * Format timestamp as relative time (e.g., "2 minutes ago")
 * @param date - Date to format
 * @returns Human-readable relative time string
 */
export function formatRelativeTime(date: Date): string {
  return formatDistanceToNow(date, { addSuffix: true });
}
