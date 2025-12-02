import type { CrowdStatus } from '../types';

interface CrowdStatusBadgeProps {
  status: CrowdStatus;
  count: number;
}

export function CrowdStatusBadge({ status, count }: CrowdStatusBadgeProps) {
  const getStatusStyles = () => {
    switch (status) {
      case 'low':
        return 'bg-green-100 text-green-800 border-green-200';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'high':
        return 'bg-red-100 text-red-800 border-red-200';
    }
  };

  const getStatusLabel = () => {
    switch (status) {
      case 'low':
        return 'Low Density';
      case 'medium':
        return 'Medium Density';
      case 'high':
        return 'High Density';
    }
  };

  return (
    <span
      className={`inline-flex items-center px-3 py-1 rounded-md text-sm font-medium border ${getStatusStyles()}`}
      title={`${count} people detected`}
    >
      {getStatusLabel()}
    </span>
  );
}
