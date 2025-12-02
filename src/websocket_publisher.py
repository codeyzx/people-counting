"""
WebSocket publisher for sending detection events to backend.

This module handles WebSocket connection management, event buffering,
and automatic reconnection with exponential backoff.
"""

import asyncio
import logging
import json
from collections import deque
from typing import Optional, Dict, Any
from .models import DetectionEvent, EventType


logger = logging.getLogger(__name__)


class WebSocketPublisher:
    """Publishes detection events to backend via WebSocket.
    
    This class manages WebSocket connection, handles disconnections with
    automatic reconnection using exponential backoff, and buffers events
    when connection is unavailable.
    """
    
    def __init__(
        self,
        url: str,
        buffer_size: int = 1000,
        frame_buffer_size: int = 10,
        reconnect_interval: float = 1.0,
        max_reconnect_interval: float = 60.0
    ):
        """Initialize WebSocket publisher.
        
        Args:
            url: WebSocket server URL (ws:// or wss://)
            buffer_size: Maximum number of events to buffer when disconnected
            frame_buffer_size: Maximum number of frames to buffer (frames are large)
            reconnect_interval: Initial reconnection interval in seconds
            max_reconnect_interval: Maximum reconnection interval in seconds
        """
        self.url = url
        self.buffer_size = buffer_size
        self.frame_buffer_size = frame_buffer_size
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_interval = max_reconnect_interval
        
        self.websocket = None
        self.is_connected = False
        self.event_buffer = deque(maxlen=buffer_size)
        self.frame_buffer = deque(maxlen=frame_buffer_size)
        self.reconnect_task = None
        self.current_reconnect_interval = reconnect_interval
        self._shutdown = False
        
        logger.info(f"WebSocket publisher initialized for {url}")
        logger.info(f"Event buffer: {buffer_size}, Frame buffer: {frame_buffer_size}, Reconnect interval: {reconnect_interval}s")
    
    async def connect(self) -> bool:
        """Establish WebSocket connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            import websockets
            
            logger.info(f"Connecting to WebSocket server: {self.url}")
            
            self.websocket = await websockets.connect(
                self.url,
                ping_interval=20,
                ping_timeout=10
            )
            
            self.is_connected = True
            self.current_reconnect_interval = self.reconnect_interval
            
            logger.info("WebSocket connection established")
            
            # Send buffered events
            await self._flush_buffer()
            
            return True
            
        except ImportError:
            logger.error("websockets library not installed. Install with: pip install websockets")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self) -> None:
        """Close WebSocket connection gracefully."""
        logger.info("Disconnecting WebSocket")
        
        self._shutdown = True
        
        # Cancel reconnect task if running
        if self.reconnect_task and not self.reconnect_task.done():
            self.reconnect_task.cancel()
            try:
                await self.reconnect_task
            except asyncio.CancelledError:
                pass
        
        # Close connection
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")
        
        self.is_connected = False
        self.websocket = None
        
        logger.info("WebSocket disconnected")
    
    async def publish_event(self, event: DetectionEvent) -> bool:
        """Publish detection event to WebSocket.
        
        If connection is not available, event is buffered for later transmission.
        
        Args:
            event: DetectionEvent to publish
            
        Returns:
            True if event was sent immediately, False if buffered
        """
        if not self.is_connected or not self.websocket:
            # Buffer event for later
            self._add_to_buffer(event)
            
            # Start reconnection if not already running
            if not self.reconnect_task or self.reconnect_task.done():
                self.reconnect_task = asyncio.create_task(self._reconnect_loop())
            
            return False
        
        try:
            # Convert event to JSON
            payload = self._create_event_payload(event)
            message = json.dumps(payload)
            
            # Send to WebSocket
            await self.websocket.send(message)
            
            logger.debug(f"Event published: {event.event_type.value}, frame {event.frame_number}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send event: {e}")
            
            # Mark as disconnected
            self.is_connected = False
            
            # Buffer the event
            self._add_to_buffer(event)
            
            # Start reconnection
            if not self.reconnect_task or self.reconnect_task.done():
                self.reconnect_task = asyncio.create_task(self._reconnect_loop())
            
            return False
    
    async def publish_frame(
        self,
        frame_data: str,
        source_id: str,
        frame_number: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Publish encoded video frame.
        
        Args:
            frame_data: Base64-encoded JPEG frame
            source_id: Camera/source identifier
            frame_number: Frame number
            metadata: Optional metadata (fps, resolution, etc.)
            
        Returns:
            True if frame was sent immediately, False if buffered
        """
        if not self.is_connected or not self.websocket:
            # Buffer frame for later
            self._add_frame_to_buffer(frame_data, source_id, frame_number, metadata)
            
            # Start reconnection if not already running
            if not self.reconnect_task or self.reconnect_task.done():
                self.reconnect_task = asyncio.create_task(self._reconnect_loop())
            
            return False
        
        try:
            # Create frame message
            payload = {
                "type": "frame",
                "timestamp": DetectionEvent.create_now(
                    source_id=source_id,
                    frame_number=frame_number,
                    current_count=0,
                    tracked_persons=[]
                ).timestamp,
                "source_id": source_id,
                "frame_number": frame_number,
                "frame": frame_data,
                "metadata": metadata or {}
            }
            
            message = json.dumps(payload)
            
            # Send to WebSocket
            await self.websocket.send(message)
            
            logger.debug(f"Frame published: source={source_id}, frame={frame_number}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send frame: {e}")
            
            # Mark as disconnected
            self.is_connected = False
            
            # Buffer the frame
            self._add_frame_to_buffer(frame_data, source_id, frame_number, metadata)
            
            # Start reconnection
            if not self.reconnect_task or self.reconnect_task.done():
                self.reconnect_task = asyncio.create_task(self._reconnect_loop())
            
            return False
    
    async def publish_lifecycle_event(
        self,
        lifecycle_event: str,
        source_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Publish lifecycle event (started, stopped, etc.).
        
        Args:
            lifecycle_event: Type of lifecycle event (e.g., "started", "stopped")
            source_id: Camera/source identifier
            metadata: Optional metadata
            
        Returns:
            True if event was sent immediately, False if buffered
        """
        from .models import TrackedPersonInfo
        
        # Create lifecycle event
        event = DetectionEvent.create_now(
            source_id=source_id,
            frame_number=0,
            current_count=0,
            tracked_persons=[],
            event_type=EventType.LIFECYCLE,
            metadata={
                "lifecycle_event": lifecycle_event,
                **(metadata or {})
            }
        )
        
        return await self.publish_event(event)
    
    def _add_to_buffer(self, event: DetectionEvent) -> None:
        """Add event to buffer.
        
        If buffer is full, oldest event is automatically removed (FIFO).
        
        Args:
            event: DetectionEvent to buffer
        """
        self.event_buffer.append(event)
        
        if len(self.event_buffer) >= self.buffer_size:
            logger.warning(f"Event buffer full ({self.buffer_size}), oldest events being dropped")
    
    def _add_frame_to_buffer(
        self,
        frame_data: str,
        source_id: str,
        frame_number: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add frame to buffer.
        
        If buffer is full, oldest frame is automatically removed (FIFO).
        Frames have a smaller buffer than events due to their size.
        
        Args:
            frame_data: Base64-encoded frame
            source_id: Camera/source identifier
            frame_number: Frame number
            metadata: Optional metadata
        """
        frame_entry = {
            "frame_data": frame_data,
            "source_id": source_id,
            "frame_number": frame_number,
            "metadata": metadata
        }
        
        self.frame_buffer.append(frame_entry)
        
        if len(self.frame_buffer) >= self.frame_buffer_size:
            logger.warning(f"Frame buffer full ({self.frame_buffer_size}), oldest frames being dropped")
    
    async def _flush_buffer(self) -> None:
        """Send all buffered events and frames."""
        # Flush events
        if self.event_buffer:
            logger.info(f"Flushing {len(self.event_buffer)} buffered events")
            
            # Create a copy to avoid modification during iteration
            events_to_send = list(self.event_buffer)
            self.event_buffer.clear()
            
            for event in events_to_send:
                try:
                    payload = self._create_event_payload(event)
                    message = json.dumps(payload)
                    await self.websocket.send(message)
                except Exception as e:
                    logger.error(f"Failed to send buffered event: {e}")
                    # Re-add to buffer
                    self._add_to_buffer(event)
                    break
            
            logger.info("Event buffer flush completed")
        
        # Flush frames
        if self.frame_buffer:
            logger.info(f"Flushing {len(self.frame_buffer)} buffered frames")
            
            # Create a copy to avoid modification during iteration
            frames_to_send = list(self.frame_buffer)
            self.frame_buffer.clear()
            
            for frame_entry in frames_to_send:
                try:
                    payload = {
                        "type": "frame",
                        "timestamp": DetectionEvent.create_now(
                            source_id=frame_entry["source_id"],
                            frame_number=frame_entry["frame_number"],
                            current_count=0,
                            tracked_persons=[]
                        ).timestamp,
                        "source_id": frame_entry["source_id"],
                        "frame_number": frame_entry["frame_number"],
                        "frame": frame_entry["frame_data"],
                        "metadata": frame_entry["metadata"] or {}
                    }
                    message = json.dumps(payload)
                    await self.websocket.send(message)
                except Exception as e:
                    logger.error(f"Failed to send buffered frame: {e}")
                    # Re-add to buffer
                    self._add_frame_to_buffer(
                        frame_entry["frame_data"],
                        frame_entry["source_id"],
                        frame_entry["frame_number"],
                        frame_entry["metadata"]
                    )
                    break
            
            logger.info("Frame buffer flush completed")
    
    async def _reconnect_loop(self) -> None:
        """Automatic reconnection loop with exponential backoff."""
        if self._shutdown:
            return
        
        logger.info("Starting reconnection loop")
        
        while not self.is_connected and not self._shutdown:
            logger.info(f"Attempting to reconnect in {self.current_reconnect_interval}s...")
            
            await asyncio.sleep(self.current_reconnect_interval)
            
            if self._shutdown:
                break
            
            # Try to connect
            success = await self.connect()
            
            if success:
                logger.info("Reconnection successful")
                break
            else:
                # Exponential backoff
                self.current_reconnect_interval = min(
                    self.current_reconnect_interval * 2,
                    self.max_reconnect_interval
                )
                logger.info(f"Reconnection failed, next attempt in {self.current_reconnect_interval}s")
    
    def _create_event_payload(self, event: DetectionEvent) -> Dict[str, Any]:
        """Create JSON payload from DetectionEvent.
        
        Args:
            event: DetectionEvent to serialize
            
        Returns:
            Dictionary ready for JSON serialization with type field
        """
        payload = event.to_dict()
        # Add type field for message routing
        payload["type"] = "detection"
        return payload
    
    def get_buffer_size(self) -> int:
        """Get current number of buffered events.
        
        Returns:
            Number of events in buffer
        """
        return len(self.event_buffer)
    
    def get_frame_buffer_size(self) -> int:
        """Get current number of buffered frames.
        
        Returns:
            Number of frames in buffer
        """
        return len(self.frame_buffer)
    
    def is_buffer_full(self) -> bool:
        """Check if event buffer is at capacity.
        
        Returns:
            True if buffer is full
        """
        return len(self.event_buffer) >= self.buffer_size
    
    def is_frame_buffer_full(self) -> bool:
        """Check if frame buffer is at capacity.
        
        Returns:
            True if frame buffer is full
        """
        return len(self.frame_buffer) >= self.frame_buffer_size
