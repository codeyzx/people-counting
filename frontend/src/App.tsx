import React from 'react';
import { DeviceProvider, useDeviceContext } from './hooks/useDeviceState';
import { useWebSocket } from './hooks/useWebSocket';
import { useOfflineDetection } from './hooks/useOfflineDetection';
import { ConnectionStatus } from './components/ConnectionStatus';
import { DeviceGrid } from './components/DeviceGrid';

function DashboardContent() {
  const { state, updateDevice, updateFrame, setConnectionStatus, setError } = useDeviceContext();

  // WebSocket URL - direct connection to backend
  const wsUrl = 'ws://localhost:8000/ws';
  
  console.log('='.repeat(50));
  console.log('[App] Dashboard initializing...');
  console.log('[App] WebSocket URL:', wsUrl);
  console.log('[App] Mode:', import.meta.env.MODE);
  console.log('='.repeat(50));
  
  // Memoize callbacks to prevent reconnection loops
  const handleMessage = React.useCallback((message: any) => {
    console.log('[App] Received message type:', message.type);
    // Route message based on type
    if (message.type === 'detection') {
      updateDevice(message);
    } else if (message.type === 'frame') {
      updateFrame(message);
    }
  }, [updateDevice, updateFrame]);

  const handleError = React.useCallback((error: Event) => {
    console.error('[App] WebSocket error:', error);
    setError('Connection error occurred');
  }, [setError]);

  const { connectionStatus, lastError } = useWebSocket({
    url: wsUrl,
    onMessage: handleMessage,
    onError: handleError,
  });

  // Update connection status in context
  React.useEffect(() => {
    setConnectionStatus(connectionStatus);
  }, [connectionStatus, setConnectionStatus]);

  React.useEffect(() => {
    setError(lastError);
  }, [lastError, setError]);

  // Enable offline device detection
  useOfflineDetection();

  // Convert Map to Array for DeviceGrid
  const devicesArray = Array.from(state.devices.values());

  return (
    <div className="min-h-screen bg-gray-50">
      <ConnectionStatus status={state.connectionStatus} error={state.lastError} />
      
      <div className={state.connectionStatus !== 'connected' ? 'pt-16' : ''}>
        <header className="bg-white shadow-sm">
          <div className="max-w-7xl mx-auto px-4 py-6">
            <h1 className="text-3xl font-bold text-gray-900">
              Real-time Device Monitoring Dashboard
            </h1>
            <p className="mt-2 text-sm text-gray-600">
              Monitor crowd detection with live video preview across all connected devices
            </p>
          </div>
        </header>

        <main>
          <DeviceGrid devices={devicesArray} frames={state.frames} />
        </main>
      </div>
    </div>
  );
}

function App() {
  return (
    <DeviceProvider>
      <DashboardContent />
    </DeviceProvider>
  );
}

export default App;
