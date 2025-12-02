import type { CrowdStatus } from '../types';

/**
 * Calculate crowd status based on count thresholds
 * @param count - Number of people detected
 * @returns Crowd status: 'low' (0-5), 'medium' (6-15), 'high' (>15)
 */
export function calculateCrowdStatus(count: number): CrowdStatus {
  if (count <= 5) return 'low';
  if (count <= 15) return 'medium';
  return 'high';
}
