import type { ConnectionStatus as ConnectionStatusType } from '../types';

interface ConnectionStatusProps {
  status: ConnectionStatusType;
  error?: string | null;
}

export function ConnectionStatus({ status, error }: ConnectionStatusProps) {
  const getStatusConfig = () => {
    switch (status) {
      case 'connecting':
        return {
          bg: 'bg-yellow-50 border-yellow-200',
          text: 'text-yellow-800',
          icon: 'text-yellow-400',
          message: 'Connecting to server...',
        };
      case 'connected':
        return {
          bg: 'bg-green-50 border-green-200',
          text: 'text-green-800',
          icon: 'text-green-400',
          message: 'Connected',
        };
      case 'disconnected':
        return {
          bg: 'bg-red-50 border-red-200',
          text: 'text-red-800',
          icon: 'text-red-400',
          message: 'Disconnected - Attempting to reconnect...',
        };
      case 'error':
        return {
          bg: 'bg-red-50 border-red-200',
          text: 'text-red-800',
          icon: 'text-red-400',
          message: error || 'Connection error',
        };
    }
  };

  const config = getStatusConfig();

  // Don't show banner when connected
  if (status === 'connected') {
    return null;
  }

  return (
    <div className={`fixed top-0 left-0 right-0 z-50 border-b ${config.bg}`}>
      <div className="max-w-7xl mx-auto px-4 py-3">
        <div className="flex items-center">
          <div className="flex-shrink-0">
            {status === 'connecting' ? (
              <svg
                className={`animate-spin h-5 w-5 ${config.icon}`}
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
            ) : (
              <svg
                className={`h-5 w-5 ${config.icon}`}
                fill="currentColor"
                viewBox="0 0 20 20"
              >
                <path
                  fillRule="evenodd"
                  d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                  clipRule="evenodd"
                />
              </svg>
            )}
          </div>
          <div className="ml-3">
            <p className={`text-sm font-medium ${config.text}`}>{config.message}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
