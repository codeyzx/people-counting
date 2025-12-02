"""
ROI (Region of Interest) filtering module.

This module handles filtering detections based on a user-defined polygonal
region of interest. Only detections with centroids inside the ROI polygon
are passed through to tracking.
"""

import logging
import json
from typing import List, Optional, Tuple
import numpy as np
import cv2
from .models import Detection, ROIConfig


logger = logging.getLogger(__name__)


class ROIFilter:
    """Filters detections based on Region of Interest polygon.
    
    This class implements point-in-polygon testing to filter person detections,
    keeping only those whose centroids fall within the defined ROI polygon.
    """
    
    def __init__(self, roi_points: Optional[List[Tuple[float, float]]] = None):
        """Initialize ROI filter with polygon points.
        
        Args:
            roi_points: List of (x, y) coordinates defining the ROI polygon.
                       None means no filtering (full frame).
        """
        self.roi_points = roi_points
        
        if roi_points is not None:
            # Validate polygon has at least 3 points
            if len(roi_points) < 3:
                raise ValueError(f"ROI polygon must have at least 3 points, got {len(roi_points)}")
            
            # Convert to numpy array for OpenCV
            self.roi_polygon = np.array(roi_points, dtype=np.float32)
            logger.info(f"ROIFilter initialized with {len(roi_points)} polygon points")
        else:
            self.roi_polygon = None
            logger.info("ROIFilter initialized with no ROI (full frame mode)")

    def is_point_in_roi(self, point: Tuple[float, float]) -> bool:
        """Check if a point is inside the ROI polygon.
        
        Uses OpenCV's pointPolygonTest for accurate point-in-polygon testing.
        
        Args:
            point: (x, y) coordinate to check
            
        Returns:
            True if point is inside ROI, False otherwise.
            If no ROI is defined, always returns True (full frame mode).
        """
        if self.roi_polygon is None:
            # No ROI defined, all points are valid
            return True
        
        # Use OpenCV's pointPolygonTest
        # Returns: positive (inside), negative (outside), zero (on edge)
        result = cv2.pointPolygonTest(self.roi_polygon, point, False)
        
        # Consider points on the edge as inside
        return result >= 0

    def filter_detections(self, detections: List[Detection]) -> List[Detection]:
        """Filter detections to only include those within ROI.
        
        Filters the detection list based on centroid position. Only detections
        with centroids inside the ROI polygon are returned.
        
        Args:
            detections: List of all detections from detector
            
        Returns:
            List of detections with centroids inside ROI.
            If no ROI is defined, returns all detections unchanged.
        """
        if self.roi_polygon is None:
            # No ROI defined, return all detections
            return detections
        
        # Filter detections based on centroid position
        filtered = []
        for detection in detections:
            if self.is_point_in_roi(detection.centroid):
                filtered.append(detection)
        
        logger.debug(f"Filtered {len(detections)} detections to {len(filtered)} within ROI")
        return filtered

    @staticmethod
    def load_from_file(file_path: str, frame_width: Optional[int] = None, 
                       frame_height: Optional[int] = None) -> 'ROIFilter':
        """Load ROI configuration from JSON file.
        
        Parses a JSON configuration file containing ROI polygon coordinates
        and creates an ROIFilter instance. Supports both absolute pixel
        coordinates and normalized (0-1) coordinates.
        
        Args:
            file_path: Path to JSON configuration file
            frame_width: Frame width for normalized coordinate conversion (optional)
            frame_height: Frame height for normalized coordinate conversion (optional)
            
        Returns:
            ROIFilter instance with loaded configuration
            
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If JSON is malformed or ROI structure is invalid
            json.JSONDecodeError: If JSON parsing fails
        """
        try:
            logger.info(f"Loading ROI configuration from {file_path}")
            
            with open(file_path, 'r') as f:
                config_data = json.load(f)
            
            # Extract ROI points
            if 'roi_points' not in config_data:
                raise ValueError("ROI configuration must contain 'roi_points' field")
            
            roi_points = config_data['roi_points']
            coordinate_type = config_data.get('coordinate_type', 'absolute')
            description = config_data.get('description', '')
            
            # Convert to list of tuples
            roi_points = [(float(x), float(y)) for x, y in roi_points]
            
            # Validate polygon has at least 3 points
            if len(roi_points) < 3:
                raise ValueError(f"ROI polygon must have at least 3 points, got {len(roi_points)}")
            
            # Handle normalized coordinates
            if coordinate_type == 'normalized':
                if frame_width is None or frame_height is None:
                    logger.warning("Normalized coordinates require frame dimensions. "
                                 "Using coordinates as-is (may need conversion later)")
                else:
                    # Convert normalized (0-1) to absolute pixels
                    roi_points = [
                        (x * frame_width, y * frame_height) 
                        for x, y in roi_points
                    ]
                    logger.info(f"Converted normalized coordinates to absolute pixels")
            
            logger.info(f"Loaded ROI: {description if description else 'No description'}")
            logger.info(f"ROI has {len(roi_points)} points, coordinate type: {coordinate_type}")
            
            return ROIFilter(roi_points)
            
        except FileNotFoundError:
            logger.error(f"ROI configuration file not found: {file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from {file_path}: {e}")
            raise ValueError(f"Invalid JSON in ROI configuration file: {e}")
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Invalid ROI configuration structure: {e}")
            raise ValueError(f"Invalid ROI configuration: {e}")
