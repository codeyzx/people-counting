"""
Person detection module using ONNX Runtime.

This module provides optimized person detection using ONNX models,
designed for efficient inference on ARM processors like Orange Pi 5.
"""

import logging
from typing import List, Tuple
import numpy as np
from .models import Detection


logger = logging.getLogger(__name__)


class ONNXPersonDetector:
    """Handles person detection using ONNX Runtime for optimized inference.
    
    This class provides an optimized alternative to PyTorch-based detection,
    using ONNX Runtime for better performance on embedded devices like Orange Pi 5.
    """
    
    # COCO dataset class ID for person
    PERSON_CLASS_ID = 0
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        providers: List[str] = None,
        inter_op_num_threads: int = 4,
        intra_op_num_threads: int = 4
    ):
        """Initialize the ONNX detector.
        
        Args:
            model_path: Path to ONNX model file
            confidence_threshold: Minimum confidence for valid detections (0.0 to 1.0)
            providers: List of execution providers (e.g., ['CPUExecutionProvider'])
            inter_op_num_threads: Number of threads for inter-op parallelism
            intra_op_num_threads: Number of threads for intra-op parallelism
            
        Raises:
            FileNotFoundError: If model file does not exist
            Exception: If model fails to load
        """
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        
        try:
            import onnxruntime as ort
            
            # Default to CPU provider if not specified
            if providers is None:
                providers = ['CPUExecutionProvider']
            
            logger.info(f"Loading ONNX model from {model_path}")
            logger.info(f"Providers: {providers}")
            logger.info(f"Threads - inter_op: {inter_op_num_threads}, intra_op: {intra_op_num_threads}")
            
            # Configure session options
            sess_options = ort.SessionOptions()
            sess_options.inter_op_num_threads = inter_op_num_threads
            sess_options.intra_op_num_threads = intra_op_num_threads
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Create inference session
            self.session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=providers
            )
            
            # Get model input/output info
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            # Extract input dimensions
            self.input_height = self.input_shape[2] if len(self.input_shape) > 2 else 640
            self.input_width = self.input_shape[3] if len(self.input_shape) > 3 else 640
            
            logger.info(f"ONNX model loaded successfully")
            logger.info(f"Input: {self.input_name}, shape: {self.input_shape}")
            logger.info(f"Input size: {self.input_height}x{self.input_width}")
            logger.info(f"Outputs: {len(self.output_names)} tensors")
            logger.info(f"Active providers: {self.session.get_providers()}")
            
        except ImportError:
            logger.error("onnxruntime not installed. Install with: pip install onnxruntime")
            raise ImportError("onnxruntime is required for ONNXPersonDetector")
        except FileNotFoundError:
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Error: Model file '{model_path}' not found")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise Exception(f"Error: Failed to load ONNX model - {e}")
    
    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Preprocess frame for ONNX model inference.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            
        Returns:
            Tuple of (preprocessed_frame, original_shape)
            - preprocessed_frame: Normalized and resized frame ready for inference
            - original_shape: Original frame dimensions (height, width)
        """
        import cv2
        
        original_shape = frame.shape[:2]  # (height, width)
        
        # Resize to model input size
        resized = cv2.resize(frame, (self.input_width, self.input_height))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Transpose from HWC to CHW format
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)
        
        return batched, original_shape
    
    def _postprocess(
        self,
        outputs: List[np.ndarray],
        original_shape: Tuple[int, int]
    ) -> List[Detection]:
        """Postprocess ONNX model outputs to extract person detections.
        
        Args:
            outputs: Raw outputs from ONNX model
            original_shape: Original frame dimensions (height, width)
            
        Returns:
            List of Detection objects
        """
        detections = []
        
        # YOLOv8 ONNX output format: [batch, 84, num_boxes]
        # First 4 values are bbox (x_center, y_center, width, height)
        # Remaining 80 values are class probabilities
        output = outputs[0]
        
        # Transpose to [batch, num_boxes, 84]
        if len(output.shape) == 3:
            output = np.transpose(output, (0, 2, 1))
        
        # Get predictions for first batch
        predictions = output[0]
        
        orig_height, orig_width = original_shape
        scale_x = orig_width / self.input_width
        scale_y = orig_height / self.input_height
        
        for pred in predictions:
            # Extract bbox and class scores
            x_center, y_center, width, height = pred[:4]
            class_scores = pred[4:]
            
            # Get class with highest score
            class_id = np.argmax(class_scores)
            confidence = float(class_scores[class_id])
            
            # Filter: only person class and above confidence threshold
            if class_id == self.PERSON_CLASS_ID and confidence >= self.confidence_threshold:
                # Convert from center format to corner format
                x1 = (x_center - width / 2) * scale_x
                y1 = (y_center - height / 2) * scale_y
                x2 = (x_center + width / 2) * scale_x
                y2 = (y_center + height / 2) * scale_y
                
                # Clip to frame boundaries
                x1 = max(0, min(x1, orig_width))
                y1 = max(0, min(y1, orig_height))
                x2 = max(0, min(x2, orig_width))
                y2 = max(0, min(y2, orig_height))
                
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
        
        # Apply Non-Maximum Suppression
        detections = self._apply_nms(detections, iou_threshold=0.45)
        
        return detections
    
    def _apply_nms(
        self,
        detections: List[Detection],
        iou_threshold: float = 0.45
    ) -> List[Detection]:
        """Apply Non-Maximum Suppression to remove overlapping detections.
        
        Args:
            detections: List of Detection objects
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Filtered list of Detection objects
        """
        if len(detections) == 0:
            return []
        
        # Extract bounding boxes and scores
        boxes = np.array([d.bbox for d in detections])
        scores = np.array([d.confidence for d in detections])
        
        # Calculate areas
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort by confidence score
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # Calculate IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            
            # Keep boxes with IoU less than threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return [detections[i] for i in keep]
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect people in a frame using ONNX Runtime.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            
        Returns:
            List of Detection objects containing bounding boxes and confidence scores
        """
        try:
            # Preprocess frame
            input_tensor, original_shape = self._preprocess(frame)
            
            # Run inference
            outputs = self.session.run(
                self.output_names,
                {self.input_name: input_tensor}
            )
            
            # Postprocess outputs
            detections = self._postprocess(outputs, original_shape)
            
            logger.debug(f"Detected {len(detections)} people in frame")
            return detections
            
        except Exception as e:
            logger.error(f"ONNX detection failed: {e}")
            # Return empty list on error to allow processing to continue
            return []
