"""
Data models for person detection and tracking system.

This module defines the core data structures used throughout the application.
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional


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
