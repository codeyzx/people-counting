"""
Frame encoder for video streaming.

This module provides frame encoding functionality for WebSocket transmission,
including JPEG compression, base64 encoding, and optional frame resizing.
"""

import base64
import logging
from typing import Optional
import cv2
import numpy as np


logger = logging.getLogger(__name__)


class FrameEncoder:
    """Encodes video frames for WebSocket transmission.
    
    This class handles frame encoding with JPEG compression, base64 encoding,
    and optional resizing to optimize bandwidth usage while maintaining
    acceptable visual quality.
    """
    
    def __init__(
        self,
        quality: int = 85,
        max_width: Optional[int] = None,
        max_height: Optional[int] = None
    ):
        """Initialize frame encoder.
        
        Args:
            quality: JPEG quality (1-100), higher is better quality but larger size
            max_width: Optional maximum width for resizing (maintains aspect ratio)
            max_height: Optional maximum height for resizing (maintains aspect ratio)
            
        Raises:
            ValueError: If quality is not in range 1-100
        """
        if not 1 <= quality <= 100:
            raise ValueError(f"Quality must be between 1 and 100, got {quality}")
        
        self.quality = quality
        self.max_width = max_width
        self.max_height = max_height
        
        # JPEG encoding parameters
        self.encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        
        logger.info(f"FrameEncoder initialized: quality={quality}, max_width={max_width}, max_height={max_height}")
    
    def encode_frame(self, frame: np.ndarray) -> str:
        """Encode frame as base64 JPEG string.
        
        Args:
            frame: OpenCV frame (BGR format) as numpy array
            
        Returns:
            Base64-encoded JPEG string
            
        Raises:
            ValueError: If frame is invalid or encoding fails
        """
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame: frame is None or empty")
        
        try:
            # Resize frame if max dimensions specified
            if self.max_width is not None or self.max_height is not None:
                frame = self._resize_frame(frame)
            
            # Encode frame as JPEG
            success, encoded_image = cv2.imencode('.jpg', frame, self.encode_params)
            
            if not success:
                raise ValueError("Failed to encode frame as JPEG")
            
            # Convert to base64
            base64_string = base64.b64encode(encoded_image).decode('utf-8')
            
            logger.debug(f"Frame encoded: size={len(base64_string)} bytes")
            
            return base64_string
            
        except Exception as e:
            logger.error(f"Frame encoding failed: {e}")
            raise ValueError(f"Frame encoding failed: {e}")
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to fit within max dimensions while maintaining aspect ratio.
        
        Args:
            frame: Original frame
            
        Returns:
            Resized frame
        """
        height, width = frame.shape[:2]
        
        # Calculate scaling factor
        scale = 1.0
        
        if self.max_width is not None and width > self.max_width:
            scale = min(scale, self.max_width / width)
        
        if self.max_height is not None and height > self.max_height:
            scale = min(scale, self.max_height / height)
        
        # Only resize if scaling is needed
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            logger.debug(f"Frame resized: {width}x{height} -> {new_width}x{new_height}")
            
            return resized
        
        return frame
    
    def get_encoded_size(self, frame: np.ndarray) -> int:
        """Get the size of encoded frame in bytes.
        
        Useful for bandwidth estimation and monitoring.
        
        Args:
            frame: Frame to encode
            
        Returns:
            Size in bytes of base64-encoded JPEG
        """
        try:
            encoded = self.encode_frame(frame)
            return len(encoded)
        except Exception as e:
            logger.error(f"Failed to get encoded size: {e}")
            return 0
    
    def set_quality(self, quality: int) -> None:
        """Update JPEG quality setting.
        
        Args:
            quality: New JPEG quality (1-100)
            
        Raises:
            ValueError: If quality is not in range 1-100
        """
        if not 1 <= quality <= 100:
            raise ValueError(f"Quality must be between 1 and 100, got {quality}")
        
        self.quality = quality
        self.encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        
        logger.info(f"JPEG quality updated to {quality}")
