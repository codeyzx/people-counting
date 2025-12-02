/**
 * Check if device should be considered offline
 * @param lastSeen - Last seen timestamp
 * @param thresholdMs - Offline threshold in milliseconds (default: 30000ms = 30s)
 * @returns True if device is offline
 */
export function isDeviceOffline(lastSeen: Date, thresholdMs: number = 30000): boolean {
  const now = Date.now();
  const timeSinceLastSeen = now - lastSeen.getTime();
  return timeSinceLastSeen > thresholdMs;
}
