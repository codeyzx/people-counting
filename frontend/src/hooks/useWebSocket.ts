import { useEffect, useRef, useState, useCallback } from 'react';
import type { WebSocketMessage, DetectionMessage, FrameMessage, ConnectionStatus } from '../types';

interface UseWebSocketOptions {
  url: string;
  onMessage: (message: WebSocketMessage) => void;
  onError?: (error: Event) => void;
}

export function useWebSocket({ url, onMessage, onError }: UseWebSocketOptions) {
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('connecting');
  const [lastError, setLastError] = useState<string | null>(null);
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttempts = useRef<number>(0);
  const shouldReconnect = useRef<boolean>(true);
  const reconnectTimeoutRef = useRef<number | undefined>(undefined);

  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const data = JSON.parse(event.data);
      
      // Validate basic fields
      if (!data.source_id || !data.timestamp) {
        console.error('[WebSocket] Invalid message format: missing source_id or timestamp');
        return;
      }
      
      // Route based on message type
      const messageType = data.type || (data.frame ? 'frame' : 'detection');
      
      if (messageType === 'frame') {
        // Frame message
        const frameMessage: FrameMessage = {
          type: 'frame',
          timestamp: data.timestamp,
          source_id: data.source_id,
          frame_number: data.frame_number || 0,
          frame: data.frame || '',
          metadata: data.metadata
        };
        
        if (!frameMessage.frame) {
          console.error('[WebSocket] Invalid frame message: missing frame data');
          return;
        }
        
        console.log(`[WebSocket] 📹 Frame from ${data.source_id}`);
        onMessage(frameMessage);
        
      } else if (messageType === 'detection') {
        // Detection message
        const detectionMessage: DetectionMessage = {
          type: 'detection',
          timestamp: data.timestamp,
          source_id: data.source_id,
          frame_number: data.frame_number || 0,
          event_type: data.event_type || 'update',
          current_count: data.current_count || 0,
          tracked_persons: data.tracked_persons || [],
          metadata: data.metadata
        };
        
        console.log(`[WebSocket] 👥 Detection from ${data.source_id}: ${data.current_count} people`);
        onMessage(detectionMessage);
        
      } else {
        console.warn('[WebSocket] Unknown message type:', messageType);
      }
      
    } catch (error) {
      console.error('[WebSocket] Failed to parse message:', error);
    }
  }, [onMessage]);

  useEffect(() => {
    console.log('[WebSocket] Initializing connection to:', url);
    shouldReconnect.current = true;
    
    const connect = () => {
      if (!shouldReconnect.current) {
        console.log('[WebSocket] Skipping connect (unmounted)');
        return;
      }
      
      // Clean up existing connection
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      
      console.log('[WebSocket] Creating new WebSocket connection...');
      setConnectionStatus('connecting');
      
      try {
        const ws = new WebSocket(url);
        wsRef.current = ws;
        
        ws.onopen = () => {
          console.log('[WebSocket] ✅ Connected successfully!');
          setConnectionStatus('connected');
          setLastError(null);
          reconnectAttempts.current = 0;
        };
        
        ws.onmessage = handleMessage;
        
        ws.onerror = (error) => {
          console.error('[WebSocket] ❌ Error:', error);
          setConnectionStatus('error');
          setLastError('Connection error');
          if (onError) {
            onError(error);
          }
        };
        
        ws.onclose = (event) => {
          console.log(`[WebSocket] 🔌 Closed: code=${event.code}, reason=${event.reason || 'none'}`);
          wsRef.current = null;
          
          if (!shouldReconnect.current) {
            console.log('[WebSocket] Not reconnecting (unmounted)');
            return;
          }
          
          setConnectionStatus('disconnected');
          
          // Exponential backoff: 1s, 2s, 4s, 8s, 16s, 32s, 60s (max)
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 60000);
          reconnectAttempts.current++;
          
          console.log(`[WebSocket] Reconnecting in ${delay}ms (attempt ${reconnectAttempts.current})...`);
          
          reconnectTimeoutRef.current = window.setTimeout(() => {
            if (shouldReconnect.current) {
              connect();
            }
          }, delay);
        };
        
      } catch (error) {
        console.error('[WebSocket] Failed to create WebSocket:', error);
        setConnectionStatus('error');
        setLastError('Failed to create connection');
        
        // Retry
        const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 60000);
        reconnectAttempts.current++;
        
        reconnectTimeoutRef.current = window.setTimeout(() => {
          if (shouldReconnect.current) {
            connect();
          }
        }, delay);
      }
    };
    
    // Start connection after small delay
    const initialTimer = window.setTimeout(connect, 100);
    
    // Cleanup
    return () => {
      console.log('[WebSocket] Cleaning up...');
      shouldReconnect.current = false;
      
      clearTimeout(initialTimer);
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [url, handleMessage, onError]);

  return {
    connectionStatus,
    lastError,
  };
}
