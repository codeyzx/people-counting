import type { DeviceInfo } from '../types';
import { DeviceCard } from './DeviceCard';
import { EmptyState } from './EmptyState';

interface DeviceGridProps {
  devices: DeviceInfo[];
  frames: Map<string, string>;
}

export function DeviceGrid({ devices, frames }: DeviceGridProps) {
  if (devices.length === 0) {
    return <EmptyState />;
  }

  // Sort devices: online first, then by source_id
  const sortedDevices = [...devices].sort((a, b) => {
    if (a.status === b.status) {
      return a.source_id.localeCompare(b.source_id);
    }
    return a.status === 'online' ? -1 : 1;
  });

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 p-6">
      {sortedDevices.map((device) => (
        <DeviceCard 
          key={device.source_id} 
          device={device}
          frameData={frames.get(device.source_id) || null}
        />
      ))}
    </div>
  );
}
