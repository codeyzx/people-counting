"""
Video processing pipeline orchestrator.

This module coordinates the detection, tracking, and rendering components
to process video streams from webcam or file inputs.
"""

import logging
import time
import asyncio
from typing import Union, Optional, List, Set
import cv2
from .detector import PersonDetector
from .tracker import PersonTracker
from .renderer import FrameRenderer
from .roi_filter import ROIFilter
from .models import DetectionEvent, TrackedPersonInfo, TrackedPerson, EventType


logger = logging.getLogger(__name__)


class VideoProcessor:
    """Main video processing pipeline orchestrator.
    
    Coordinates the detection, tracking, and rendering components to process
    video frames from webcam or file inputs.
    """
    
    def __init__(
        self,
        detector: PersonDetector,
        tracker: PersonTracker,
        renderer: FrameRenderer,
        roi_filter: Optional[ROIFilter] = None,
        websocket_publisher=None,
        source_id: str = "camera_01"
    ):
        """Initialize the processor with required components.
        
        Args:
            detector: PersonDetector instance for detecting people
            tracker: PersonTracker instance for tracking people across frames
            renderer: FrameRenderer instance for visualizing results
            roi_filter: Optional ROIFilter instance for filtering detections by region
            websocket_publisher: Optional WebSocketPublisher for sending events
            source_id: Unique identifier for this camera/source
        """
        self.detector = detector
        self.tracker = tracker
        self.renderer = renderer
        self.roi_filter = roi_filter
        self.websocket_publisher = websocket_publisher
        self.source_id = source_id
        
        # Track previous state for detecting changes
        self.previous_count = 0
        self.previous_person_ids: Set[int] = set()
        
        if roi_filter is not None:
            logger.info("VideoProcessor initialized with ROI filtering enabled")
        else:
            logger.info("VideoProcessor initialized (full frame mode)")
        
        if websocket_publisher is not None:
            logger.info(f"VideoProcessor initialized with WebSocket publishing (source: {source_id})")
        else:
            logger.info("VideoProcessor initialized without WebSocket publishing")
    
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

    
    def _create_detection_event(
        self,
        frame_number: int,
        tracked_persons: List[TrackedPerson],
        event_type: EventType = EventType.UPDATE,
        fps: Optional[float] = None,
        inference_time: Optional[float] = None
    ) -> DetectionEvent:
        """Create detection event from tracking results.
        
        Args:
            frame_number: Current frame number
            tracked_persons: List of tracked persons
            event_type: Type of event
            fps: Current FPS (optional)
            inference_time: Inference time in ms (optional)
            
        Returns:
            DetectionEvent ready for transmission
        """
        # Convert TrackedPerson to TrackedPersonInfo
        tracked_info = [
            TrackedPersonInfo(
                person_id=p.person_id,
                bbox=p.bbox,
                confidence=p.confidence,
                centroid=p.centroid
            )
            for p in tracked_persons
        ]
        
        # Build metadata
        metadata = {}
        if fps is not None:
            metadata['fps'] = round(fps, 2)
        if inference_time is not None:
            metadata['inference_time_ms'] = round(inference_time, 2)
        
        # Create event
        event = DetectionEvent.create_now(
            source_id=self.source_id,
            frame_number=frame_number,
            current_count=len(tracked_persons),
            tracked_persons=tracked_info,
            event_type=event_type,
            metadata=metadata if metadata else None
        )
        
        return event
    
    def _detect_entry_exit_events(
        self,
        current_person_ids: Set[int],
        frame_number: int,
        tracked_persons: List[TrackedPerson]
    ) -> List[DetectionEvent]:
        """Detect person entry and exit events.
        
        Args:
            current_person_ids: Set of current person IDs
            frame_number: Current frame number
            tracked_persons: List of tracked persons
            
        Returns:
            List of entry/exit events
        """
        events = []
        
        # Detect new entries
        new_ids = current_person_ids - self.previous_person_ids
        for person_id in new_ids:
            # Find the person
            person = next((p for p in tracked_persons if p.person_id == person_id), None)
            if person:
                event = self._create_detection_event(
                    frame_number=frame_number,
                    tracked_persons=[person],
                    event_type=EventType.ENTRY
                )
                events.append(event)
                logger.debug(f"Person {person_id} entered frame")
        
        # Detect exits
        exited_ids = self.previous_person_ids - current_person_ids
        for person_id in exited_ids:
            # Create event with empty tracked persons (person has left)
            event = DetectionEvent.create_now(
                source_id=self.source_id,
                frame_number=frame_number,
                current_count=0,
                tracked_persons=[],
                event_type=EventType.EXIT,
                metadata={"exited_person_id": person_id}
            )
            events.append(event)
            logger.debug(f"Person {person_id} exited frame")
        
        return events
    
    async def process_video_async(
        self,
        input_source: Union[int, str],
        output_path: Optional[str] = None
    ) -> int:
        """Process video asynchronously with WebSocket support.
        
        Args:
            input_source: Webcam device index (int) or video file path (str)
            output_path: Optional path for output video file
            
        Returns:
            Final count of people in the last frame
        """
        cap = None
        out = None
        
        try:
            # Send lifecycle start event
            if self.websocket_publisher:
                await self.websocket_publisher.publish_lifecycle_event(
                    lifecycle_event="started",
                    source_id=self.source_id,
                    metadata={
                        "input_source": str(input_source),
                        "output_path": output_path
                    }
                )
            
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
            last_fps_log = start_time
            
            logger.info("Starting video processing... Press ESC to quit")
            
            # Main processing loop
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    logger.info("End of video stream")
                    break
                
                frame_count += 1
                frame_start = time.time()
                
                # Detect people in frame
                detections = self.detector.detect(frame)
                
                # Apply ROI filtering if enabled
                if self.roi_filter is not None:
                    detections = self.roi_filter.filter_detections(detections)
                
                # Update tracker with detections
                tracked_people = self.tracker.update(detections)
                
                # Get current count
                current_count = self.tracker.get_current_count()
                
                # Calculate metrics
                frame_time = time.time() - frame_start
                inference_time_ms = frame_time * 1000
                current_fps = 1.0 / frame_time if frame_time > 0 else 0
                
                # WebSocket event publishing
                if self.websocket_publisher:
                    current_person_ids = {p.person_id for p in tracked_people}
                    
                    # Detect and send entry/exit events
                    entry_exit_events = self._detect_entry_exit_events(
                        current_person_ids,
                        frame_count,
                        tracked_people
                    )
                    
                    for event in entry_exit_events:
                        await self.websocket_publisher.publish_event(event)
                    
                    # Send update event if count changed
                    if current_count != self.previous_count:
                        update_event = self._create_detection_event(
                            frame_number=frame_count,
                            tracked_persons=tracked_people,
                            event_type=EventType.UPDATE,
                            fps=current_fps,
                            inference_time=inference_time_ms
                        )
                        await self.websocket_publisher.publish_event(update_event)
                    
                    # Update previous state
                    self.previous_count = current_count
                    self.previous_person_ids = current_person_ids
                
                # Get ROI points for visualization
                roi_points = self.roi_filter.roi_points if self.roi_filter is not None else None
                
                # Render visualizations
                output_frame = self.renderer.render(frame, tracked_people, current_count, roi_points)
                
                # Display frame
                cv2.imshow("Person Detection & Counting", output_frame)
                
                # Write frame to output video if specified
                if out is not None:
                    out.write(output_frame)
                
                # Log FPS periodically
                current_time = time.time()
                if current_time - last_fps_log >= 5.0:  # Log every 5 seconds
                    elapsed = current_time - start_time
                    avg_fps = frame_count / elapsed if elapsed > 0 else 0
                    logger.info(f"Processing: Frame {frame_count}, FPS: {avg_fps:.2f}, Count: {current_count}")
                    last_fps_log = current_time
                
                # Check for ESC key press
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    logger.info("ESC key pressed, terminating...")
                    break
                
                # Allow other async tasks to run
                await asyncio.sleep(0)
            
            # Calculate final statistics
            end_time = time.time()
            elapsed_time = end_time - start_time
            final_count = self.tracker.get_current_count()
            avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            logger.info(f"Processing complete:")
            logger.info(f"  Frames processed: {frame_count}")
            logger.info(f"  Final count in last frame: {final_count}")
            logger.info(f"  Processing time: {elapsed_time:.2f} seconds")
            logger.info(f"  Average FPS: {avg_fps:.2f}")
            
            # Send lifecycle stop event
            if self.websocket_publisher:
                await self.websocket_publisher.publish_lifecycle_event(
                    lifecycle_event="stopped",
                    source_id=self.source_id,
                    metadata={
                        "frames_processed": frame_count,
                        "average_fps": round(avg_fps, 2),
                        "final_count": final_count
                    }
                )
            
            return final_count
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            
            # Send error lifecycle event
            if self.websocket_publisher:
                try:
                    await self.websocket_publisher.publish_lifecycle_event(
                        lifecycle_event="error",
                        source_id=self.source_id,
                        metadata={"error": str(e)}
                    )
                except:
                    pass
            
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
