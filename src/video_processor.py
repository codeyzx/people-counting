"""
Video processing pipeline orchestrator.

This module coordinates the detection, tracking, and rendering components
to process video streams from webcam or file inputs.
"""

import logging
import time
from typing import Union, Optional
import cv2
from .detector import PersonDetector
from .tracker import PersonTracker
from .renderer import FrameRenderer
from .roi_filter import ROIFilter


logger = logging.getLogger(__name__)


class VideoProcessor:
    """Main video processing pipeline orchestrator.
    
    Coordinates the detection, tracking, and rendering components to process
    video frames from webcam or file inputs.
    """
    
    def __init__(self, detector: PersonDetector, tracker: PersonTracker, 
                 renderer: FrameRenderer, roi_filter: Optional[ROIFilter] = None):
        """Initialize the processor with required components.
        
        Args:
            detector: PersonDetector instance for detecting people
            tracker: PersonTracker instance for tracking people across frames
            renderer: FrameRenderer instance for visualizing results
            roi_filter: Optional ROIFilter instance for filtering detections by region
        """
        self.detector = detector
        self.tracker = tracker
        self.renderer = renderer
        self.roi_filter = roi_filter
        
        if roi_filter is not None:
            logger.info("VideoProcessor initialized with ROI filtering enabled")
        else:
            logger.info("VideoProcessor initialized (full frame mode)")
    
    def process_video(self, input_source: Union[int, str], 
                     output_path: Optional[str] = None) -> int:
        """Process video from input source.
        
        Opens the video source (webcam or file), processes each frame through
        the detection, tracking, and rendering pipeline, and optionally saves
        the output video.
        
        Args:
            input_source: Webcam device index (int) or video file path (str)
            output_path: Optional path for output video file
            
        Returns:
            Final count of people in the last frame
            
        Raises:
            Exception: If video source cannot be opened or processing fails
        """
        cap = None
        out = None
        
        try:
            # Open video source
            logger.info(f"Opening video source: {input_source}")
            cap = cv2.VideoCapture(input_source)
            
            if not cap.isOpened():
                if isinstance(input_source, int):
                    raise Exception(f"Error: Webcam device {input_source} not available")
                else:
                    raise Exception(f"Error: Cannot open video source '{input_source}'")
            
            # Get video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            if fps == 0:
                fps = 30  # Default FPS for webcam
            
            logger.info(f"Video properties: {frame_width}x{frame_height} @ {fps} FPS")
            
            # Set up video writer if output path is specified
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
                
                if not out.isOpened():
                    raise Exception(f"Error: Cannot create output file '{output_path}'")
                
                logger.info(f"Output video will be saved to: {output_path}")
            
            # Processing statistics
            frame_count = 0
            start_time = time.time()
            
            logger.info("Starting video processing... Press ESC to quit")
            
            # Main processing loop
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    logger.info("End of video stream")
                    break
                
                frame_count += 1
                
                # Detect people in frame
                detections = self.detector.detect(frame)
                
                # Apply ROI filtering if enabled
                if self.roi_filter is not None:
                    detections = self.roi_filter.filter_detections(detections)
                
                # Update tracker with detections
                tracked_people = self.tracker.update(detections)
                
                # Get current count (real-time count of people in frame)
                current_count = self.tracker.get_current_count()
                
                # Get ROI points for visualization
                roi_points = self.roi_filter.roi_points if self.roi_filter is not None else None
                
                # Render visualizations
                output_frame = self.renderer.render(frame, tracked_people, current_count, roi_points)
                
                # Display frame
                cv2.imshow("Person Detection & Counting", output_frame)
                
                # Write frame to output video if specified
                if out is not None:
                    out.write(output_frame)
                
                # Check for ESC key press (27 is ASCII code for ESC)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    logger.info("ESC key pressed, terminating...")
                    break
            
            # Calculate processing statistics
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            final_count = self.tracker.get_current_count()
            
            logger.info(f"Processing complete:")
            logger.info(f"  Frames processed: {frame_count}")
            logger.info(f"  Final count in last frame: {final_count}")
            logger.info(f"  Processing time: {elapsed_time:.2f} seconds")
            
            if frame_count > 0:
                logger.info(f"  Average FPS: {frame_count / elapsed_time:.2f}")
            
            return final_count
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            raise
            
        finally:
            # Clean up resources
            if cap is not None:
                cap.release()
                logger.debug("Video capture released")
            
            if out is not None:
                out.release()
                logger.debug("Video writer released")
            
            cv2.destroyAllWindows()
            logger.debug("Windows destroyed")
