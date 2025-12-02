export type EventType = 'update' | 'entry' | 'exit' | 'lifecycle';

export type DeviceStatus = 'online' | 'offline';

export type CrowdStatus = 'low' | 'medium' | 'high';

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

export type MessageType = 'detection' | 'frame';

export interface TrackedPerson {
  person_id: number;
  bbox: [number, number, number, number];
  confidence: number;
  centroid: [number, number];
}

export interface DetectionMessage {
  type: 'detection';
  timestamp: string;
  source_id: string;
  frame_number: number;
  event_type: EventType;
  current_count: number;
  tracked_persons: TrackedPerson[];
  metadata?: Record<string, any>;
}

export interface FrameMessage {
  type: 'frame';
  timestamp: string;
  source_id: string;
  frame_number: number;
  frame: string; // Base64-encoded JPEG
  metadata?: Record<string, any>;
}

export type WebSocketMessage = DetectionMessage | FrameMessage;

// Legacy message format for backward compatibility
export interface LegacyWebSocketMessage {
  timestamp: string;
  source_id: string;
  frame_number: number;
  event_type: EventType;
  current_count: number;
  tracked_persons: TrackedPerson[];
  metadata?: Record<string, any>;
}

export interface DeviceInfo {
  source_id: string;
  current_count: number;
  crowd_status: CrowdStatus;
  last_seen: Date;
  status: DeviceStatus;
  frame_number: number;
  metadata?: Record<string, any>;
}
