import type { DeviceStatus } from '../types';

interface StatusBadgeProps {
  status: DeviceStatus;
}

export function StatusBadge({ status }: StatusBadgeProps) {
  const isOnline = status === 'online';
  
  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
        isOnline
          ? 'bg-green-100 text-green-800'
          : 'bg-gray-100 text-gray-800'
      }`}
    >
      <span
        className={`w-2 h-2 mr-1.5 rounded-full ${
          isOnline ? 'bg-green-400' : 'bg-gray-400'
        }`}
      />
      {isOnline ? 'Online' : 'Offline'}
    </span>
  );
}
