import { useEffect } from 'react';
import { useDeviceContext } from './useDeviceState';
import { isDeviceOffline } from '../utils';

const CHECK_INTERVAL = 5000; // 5 seconds
const OFFLINE_THRESHOLD = 30000; // 30 seconds

/**
 * Hook to periodically check for offline devices
 * Marks devices as offline if they haven't sent data in 30 seconds
 */
export function useOfflineDetection() {
  const { state, updateDeviceStatus } = useDeviceContext();

  useEffect(() => {
    const intervalId = setInterval(() => {
      state.devices.forEach((device) => {
        const shouldBeOffline = isDeviceOffline(device.last_seen, OFFLINE_THRESHOLD);
        
        // Only update if status changed
        if (shouldBeOffline && device.status === 'online') {
          console.log(`Device ${device.source_id} is now offline`);
          updateDeviceStatus(device.source_id, 'offline');
        } else if (!shouldBeOffline && device.status === 'offline') {
          console.log(`Device ${device.source_id} is back online`);
          updateDeviceStatus(device.source_id, 'online');
        }
      });
    }, CHECK_INTERVAL);

    return () => {
      clearInterval(intervalId);
    };
  }, [state.devices, updateDeviceStatus]);
}
