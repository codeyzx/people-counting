"""
Person detection module using YOLOv8.

This module handles loading the YOLOv8 model and detecting people in video frames.
"""

import logging
from typing import List
import numpy as np
from ultralytics import YOLO
from .models import Detection


logger = logging.getLogger(__name__)


class PersonDetector:
    """Handles person detection using YOLOv8 model.
    
    This class encapsulates YOLOv8 model loading and inference for detecting
    people in video frames. It filters detections to only include the person
    class and applies confidence thresholding.
    """
    
    # COCO dataset class ID for person
    PERSON_CLASS_ID = 0
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """Initialize the detector with YOLOv8 model.
        
        Args:
            model_path: Path to YOLOv8 model weights file
            confidence_threshold: Minimum confidence for valid detections (0.0 to 1.0)
            
        Raises:
            FileNotFoundError: If model file does not exist
            Exception: If model fails to load
        """
        self.confidence_threshold = confidence_threshold
        
        try:
            logger.info(f"Loading YOLOv8 model from {model_path}")
            self.model = YOLO(model_path)
            logger.info("YOLOv8 model loaded successfully")
        except FileNotFoundError:
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Error: Model file '{model_path}' not found")
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            raise Exception(f"Error: Failed to load model - {e}")
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect people in a frame.
        
        Processes the input frame using YOLOv8 and returns a list of person
        detections that meet the confidence threshold.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            
        Returns:
            List of Detection objects containing bounding boxes and confidence scores
            
        Raises:
            Exception: If detection fails
        """
        try:
            # Run YOLOv8 inference
            results = self.model(frame, verbose=False)
            
            detections = []
            
            # Process results
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    # Get class ID and confidence
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Filter: only person class and above confidence threshold
                    if class_id == self.PERSON_CLASS_ID and confidence >= self.confidence_threshold:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        bbox = (int(x1), int(y1), int(x2), int(y2))
                        
                        # Calculate centroid
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        centroid = (float(cx), float(cy))
                        
                        # Create Detection object
                        detection = Detection(
                            bbox=bbox,
                            confidence=confidence,
                            centroid=centroid
                        )
                        detections.append(detection)
            
            logger.debug(f"Detected {len(detections)} people in frame")
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            # Return empty list on error to allow processing to continue
            return []
