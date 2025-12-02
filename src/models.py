"""
Data models for person detection and tracking system.

This module defines the core data structures used throughout the application.
"""

from dataclasses import dataclass, field, asdict
from typing import Tuple, List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


@dataclass
class Detection:
    """Represents a single person detection in a frame.
    
    Attributes:
        bbox: Bounding box coordinates as (x1, y1, x2, y2)
        confidence: Detection confidence score (0.0 to 1.0)
        centroid: Center point of the bounding box as (cx, cy)
    """
    bbox: Tuple[int, int, int, int]
    confidence: float
    centroid: Tuple[float, float]


@dataclass
class TrackedPerson:
    """Represents a tracked person with unique ID.
    
    Attributes:
        person_id: Unique identifier for this person
        bbox: Bounding box coordinates as (x1, y1, x2, y2)
        centroid: Center point of the bounding box as (cx, cy)
        confidence: Detection confidence score (0.0 to 1.0)
    """
    person_id: int
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    confidence: float


@dataclass
class ROIConfig:
    """ROI configuration parameters.
    
    Attributes:
        roi_points: List of (x, y) coordinates defining the polygon
        coordinate_type: Either 'absolute' (pixels) or 'normalized' (0-1)
        description: Optional description of the ROI
    """
    roi_points: List[Tuple[float, float]]
    coordinate_type: str = "absolute"
    description: str = ""


@dataclass
class Config:
    """Application configuration parameters.
    
    Attributes:
        model_path: Path to YOLOv8 model weights file
        confidence_threshold: Minimum confidence for valid detections (0.0 to 1.0)
        tracking_distance: Maximum distance for matching detections across frames
        font_scale: Scale factor for text rendering
        line_thickness: Line thickness for boxes and text
        window_name: Name of the display window
        roi_file: Optional path to ROI configuration JSON file
        roi_color: Color for ROI boundary (BGR format)
        roi_thickness: Line thickness for ROI boundary
        roi_alpha: Transparency for ROI fill (0.0 to 1.0)
    """
    model_path: str = "models/yolov8s.pt"
    confidence_threshold: float = 0.5
    tracking_distance: float = 50.0
    font_scale: float = 0.6
    line_thickness: int = 2
    window_name: str = "Person Detection & Counting"
    roi_file: Optional[str] = None
    roi_color: Tuple[int, int, int] = (255, 0, 0)  # Blue (BGR)
    roi_thickness: int = 2
    roi_alpha: float = 0.3


@dataclass
class StreamingConfig:
    """Configuration for video streaming.
    
    Attributes:
        enable_streaming: Enable video frame streaming
        max_frame_rate: Maximum frame rate for streaming (FPS)
        jpeg_quality: JPEG quality for frame encoding (1-100)
        max_frame_width: Optional maximum frame width for resizing
        max_frame_height: Optional maximum frame height for resizing
        websocket_url: WebSocket server URL for device connection
        frame_buffer_size: Maximum number of frames to buffer
    """
    enable_streaming: bool = True
    max_frame_rate: float = 10.0
    jpeg_quality: int = 85
    max_frame_width: Optional[int] = 1280
    max_frame_height: Optional[int] = None
    websocket_url: str = "ws://localhost:8000/device"
    frame_buffer_size: int = 10


class EventType(Enum):
    """Types of detection events."""
    UPDATE = "update"
    ENTRY = "entry"
    EXIT = "exit"
    LIFECYCLE = "lifecycle"


@dataclass
class TrackedPersonInfo:
    """Simplified tracked person info for WebSocket transmission.
    
    Attributes:
        person_id: Unique identifier for this person
        bbox: Bounding box coordinates as (x1, y1, x2, y2)
        confidence: Detection confidence score (0.0 to 1.0)
        centroid: Center point of the bounding box as (cx, cy)
    """
    person_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    centroid: Tuple[float, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "person_id": self.person_id,
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "centroid": list(self.centroid)
        }


@dataclass
class DetectionEvent:
    """Event containing detection information for WebSocket transmission.
    
    Attributes:
        timestamp: ISO 8601 formatted timestamp
        source_id: Camera/source identifier
        frame_number: Current frame number
        current_count: Number of people currently detected
        tracked_persons: List of tracked persons in this frame
        event_type: Type of event (update, entry, exit, lifecycle)
        metadata: Optional additional metadata
    """
    timestamp: str
    source_id: str
    frame_number: int
    current_count: int
    tracked_persons: List[TrackedPersonInfo]
    event_type: EventType
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    @staticmethod
    def create_now(
        source_id: str,
        frame_number: int,
        current_count: int,
        tracked_persons: List[TrackedPersonInfo],
        event_type: EventType = EventType.UPDATE,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'DetectionEvent':
        """Create a DetectionEvent with current timestamp.
        
        Args:
            source_id: Camera/source identifier
            frame_number: Current frame number
            current_count: Number of people detected
            tracked_persons: List of tracked persons
            event_type: Type of event
            metadata: Optional metadata
            
        Returns:
            DetectionEvent with ISO 8601 timestamp
        """
        timestamp = datetime.utcnow().isoformat() + "Z"
        return DetectionEvent(
            timestamp=timestamp,
            source_id=source_id,
            frame_number=frame_number,
            current_count=current_count,
            tracked_persons=tracked_persons,
            event_type=event_type,
            metadata=metadata or {}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.
        
        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "timestamp": self.timestamp,
            "source_id": self.source_id,
            "frame_number": self.frame_number,
            "event_type": self.event_type.value,
            "current_count": self.current_count,
            "tracked_persons": [p.to_dict() for p in self.tracked_persons],
            "metadata": self.metadata
        }
