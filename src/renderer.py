"""
Frame rendering module for visualization.

This module handles drawing bounding boxes, person IDs, and count statistics
on video frames.
"""

import logging
from typing import List, Optional, Tuple
import numpy as np
import cv2
from .models import TrackedPerson


logger = logging.getLogger(__name__)


class FrameRenderer:
    """Renders visualizations on video frames.
    
    Draws bounding boxes around detected people, displays their IDs,
    and shows the current count (real-time) as an overlay.
    """
    
    def __init__(self, font_scale: float = 0.6, thickness: int = 2,
                 roi_color: Tuple[int, int, int] = (255, 0, 0),
                 roi_thickness: int = 2, roi_alpha: float = 0.3):
        """Initialize the renderer with display parameters.
        
        Args:
            font_scale: Scale factor for text rendering
            thickness: Line thickness for boxes and text
            roi_color: Color for ROI boundary (BGR format)
            roi_thickness: Line thickness for ROI boundary
            roi_alpha: Transparency for ROI fill (0.0 to 1.0)
        """
        self.font_scale = font_scale
        self.thickness = thickness
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Colors (BGR format)
        self.bbox_color = (0, 255, 0)  # Green for bounding boxes
        self.text_color = (255, 255, 255)  # White for text
        self.bg_color = (0, 0, 0)  # Black for text background
        self.count_bg_color = (0, 0, 0)  # Black for count background
        
        # ROI visualization parameters
        self.roi_color = roi_color
        self.roi_thickness = roi_thickness
        self.roi_alpha = roi_alpha
        
        logger.info(f"FrameRenderer initialized with font_scale={font_scale}, thickness={thickness}")
    
    def draw_roi(self, frame: np.ndarray, roi_points: List[Tuple[float, float]]) -> np.ndarray:
        """Draw ROI polygon on frame with semi-transparent fill.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            roi_points: List of (x, y) coordinates defining the ROI polygon
            
        Returns:
            Frame with ROI visualization drawn
        """
        if roi_points is None or len(roi_points) < 3:
            return frame
        
        # Convert points to numpy array for OpenCV
        points = np.array(roi_points, dtype=np.int32)
        
        # Create overlay for semi-transparent fill
        overlay = frame.copy()
        
        # Draw filled polygon on overlay
        cv2.fillPoly(overlay, [points], self.roi_color)
        
        # Blend overlay with original frame
        cv2.addWeighted(overlay, self.roi_alpha, frame, 1 - self.roi_alpha, 0, frame)
        
        # Draw polygon boundary
        cv2.polylines(frame, [points], isClosed=True, color=self.roi_color, 
                     thickness=self.roi_thickness)
        
        return frame
    
    def render(self, frame: np.ndarray, tracked_people: List[TrackedPerson], 
               current_count: int, roi_points: Optional[List[Tuple[float, float]]] = None) -> np.ndarray:
        """Render visualizations on frame.
        
        Draws ROI boundary (if provided), bounding boxes around each tracked person,
        displays their IDs, and shows the current count in the top-left corner.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            tracked_people: List of tracked people to visualize
            current_count: Current count of people in frame (real-time)
            roi_points: Optional list of ROI polygon points for visualization
            
        Returns:
            Frame with visualizations drawn
        """
        # Create a copy to avoid modifying the original frame
        output_frame = frame.copy()
        
        # Draw ROI if provided (background layer)
        if roi_points is not None:
            output_frame = self.draw_roi(output_frame, roi_points)
        
        # Draw bounding boxes and IDs for each tracked person
        for person in tracked_people:
            x1, y1, x2, y2 = person.bbox
            
            # Draw bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), self.bbox_color, self.thickness)
            
            # Prepare ID text
            id_text = f"ID: {person.person_id}"
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                id_text, self.font, self.font_scale, self.thickness
            )
            
            # Draw text background (above the bounding box)
            text_x = x1
            text_y = y1 - 10
            
            # Ensure text stays within frame bounds
            if text_y - text_height < 0:
                text_y = y1 + text_height + 10
            
            cv2.rectangle(
                output_frame,
                (text_x, text_y - text_height - baseline),
                (text_x + text_width, text_y + baseline),
                self.bg_color,
                -1  # Filled rectangle
            )
            
            # Draw ID text
            cv2.putText(
                output_frame,
                id_text,
                (text_x, text_y),
                self.font,
                self.font_scale,
                self.text_color,
                self.thickness
            )
        
        # Draw current count in top-left corner
        count_text = f"Current Count: {current_count}"
        
        # Get text size for background
        (count_width, count_height), count_baseline = cv2.getTextSize(
            count_text, self.font, self.font_scale * 1.2, self.thickness + 1
        )
        
        # Draw count background
        padding = 10
        cv2.rectangle(
            output_frame,
            (padding, padding),
            (padding + count_width + padding, padding + count_height + padding + count_baseline),
            self.count_bg_color,
            -1
        )
        
        # Draw count text
        cv2.putText(
            output_frame,
            count_text,
            (padding + 5, padding + count_height + 5),
            self.font,
            self.font_scale * 1.2,
            self.text_color,
            self.thickness + 1
        )
        
        return output_frame
