import { createContext, useContext, useReducer, useCallback } from 'react';
import type { ReactNode } from 'react';
import type { DeviceInfo, WebSocketMessage, DetectionMessage, FrameMessage, ConnectionStatus } from '../types';
import { calculateCrowdStatus } from '../utils';

interface DeviceState {
  devices: Map<string, DeviceInfo>;
  frames: Map<string, string>; // source_id -> base64 frame data
  connectionStatus: ConnectionStatus;
  lastError: string | null;
}

type DeviceAction =
  | { type: 'UPDATE_DEVICE'; payload: DetectionMessage }
  | { type: 'UPDATE_FRAME'; payload: FrameMessage }
  | { type: 'UPDATE_DEVICE_STATUS'; payload: { source_id: string; status: 'online' | 'offline' } }
  | { type: 'SET_CONNECTION_STATUS'; payload: ConnectionStatus }
  | { type: 'SET_ERROR'; payload: string | null };

interface DeviceContextValue {
  state: DeviceState;
  updateDevice: (message: DetectionMessage) => void;
  updateFrame: (message: FrameMessage) => void;
  updateDeviceStatus: (source_id: string, status: 'online' | 'offline') => void;
  setConnectionStatus: (status: ConnectionStatus) => void;
  setError: (error: string | null) => void;
}

const DeviceContext = createContext<DeviceContextValue | undefined>(undefined);

function deviceReducer(state: DeviceState, action: DeviceAction): DeviceState {
  switch (action.type) {
    case 'UPDATE_DEVICE': {
      const message = action.payload;
      const newDevices = new Map(state.devices);
      
      const deviceInfo: DeviceInfo = {
        source_id: message.source_id,
        current_count: message.current_count,
        crowd_status: calculateCrowdStatus(message.current_count),
        last_seen: new Date(message.timestamp),
        status: 'online',
        frame_number: message.frame_number,
        metadata: message.metadata,
      };
      
      newDevices.set(message.source_id, deviceInfo);
      
      return {
        ...state,
        devices: newDevices,
      };
    }
    
    case 'UPDATE_FRAME': {
      const message = action.payload;
      const newFrames = new Map(state.frames);
      
      // Store frame data for this device
      newFrames.set(message.source_id, message.frame);
      
      // Also update device last_seen timestamp
      const device = state.devices.get(message.source_id);
      if (device) {
        const newDevices = new Map(state.devices);
        newDevices.set(message.source_id, {
          ...device,
          last_seen: new Date(message.timestamp),
          status: 'online',
        });
        
        return {
          ...state,
          devices: newDevices,
          frames: newFrames,
        };
      }
      
      return {
        ...state,
        frames: newFrames,
      };
    }
    
    case 'UPDATE_DEVICE_STATUS': {
      const { source_id, status } = action.payload;
      const device = state.devices.get(source_id);
      
      if (!device) return state;
      
      const newDevices = new Map(state.devices);
      newDevices.set(source_id, {
        ...device,
        status,
      });
      
      return {
        ...state,
        devices: newDevices,
      };
    }
    
    case 'SET_CONNECTION_STATUS': {
      return {
        ...state,
        connectionStatus: action.payload,
      };
    }
    
    case 'SET_ERROR': {
      return {
        ...state,
        lastError: action.payload,
      };
    }
    
    default:
      return state;
  }
}

const initialState: DeviceState = {
  devices: new Map(),
  frames: new Map(),
  connectionStatus: 'connecting',
  lastError: null,
};

export function DeviceProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(deviceReducer, initialState);

  const updateDevice = useCallback((message: DetectionMessage) => {
    dispatch({ type: 'UPDATE_DEVICE', payload: message });
  }, []);

  const updateFrame = useCallback((message: FrameMessage) => {
    dispatch({ type: 'UPDATE_FRAME', payload: message });
  }, []);

  const updateDeviceStatus = useCallback((source_id: string, status: 'online' | 'offline') => {
    dispatch({ type: 'UPDATE_DEVICE_STATUS', payload: { source_id, status } });
  }, []);

  const setConnectionStatus = useCallback((status: ConnectionStatus) => {
    dispatch({ type: 'SET_CONNECTION_STATUS', payload: status });
  }, []);

  const setError = useCallback((error: string | null) => {
    dispatch({ type: 'SET_ERROR', payload: error });
  }, []);

  const value: DeviceContextValue = {
    state,
    updateDevice,
    updateFrame,
    updateDeviceStatus,
    setConnectionStatus,
    setError,
  };

  return <DeviceContext.Provider value={value}>{children}</DeviceContext.Provider>;
}

export function useDeviceContext() {
  const context = useContext(DeviceContext);
  if (context === undefined) {
    throw new Error('useDeviceContext must be used within a DeviceProvider');
  }
  return context;
}
