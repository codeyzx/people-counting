"""
Person Detection and Counting Application.

Main entry point for the person detection and counting system using YOLOv8.
Supports flexible input from webcam or video files with optional video output.
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from .models import Config
from .detector import PersonDetector
from .tracker import PersonTracker
from .renderer import FrameRenderer
from .video_processor import VideoProcessor
from .roi_filter import ROIFilter


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_arguments():
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='Person Detection and Counting System using YOLOv8',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default webcam (device 0)
  python person_counter.py
  
  # Use specific webcam device
  python person_counter.py --source 1
  
  # Process video file
  python person_counter.py --source video.mp4
  
  # Process video and save output
  python person_counter.py --source video.mp4 --output result.avi
  
  # Use custom confidence threshold
  python person_counter.py --source video.mp4 --confidence 0.7
        """
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default='0',
        help='Video source: webcam device index (0, 1, 2, ...) or video file path (default: 0)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output video file path (optional). If not specified, video will not be saved.'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Confidence threshold for detections (0.0 to 1.0, default: 0.5)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='models/yolov8s.pt',
        help='Path to YOLOv8 model weights file (default: yolov8s.pt)'
    )
    
    parser.add_argument(
        '--tracking-distance',
        type=float,
        default=50.0,
        help='Maximum distance for tracking people across frames (default: 50.0)'
    )
    
    parser.add_argument(
        '--roi',
        type=str,
        default=None,
        help='Path to ROI configuration JSON file (optional). If not specified, full frame detection is used.'
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate command-line arguments.
    
    Args:
        args: Parsed arguments namespace
        
    Raises:
        ValueError: If arguments are invalid
    """
    # Validate confidence threshold
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("Error: Confidence threshold must be between 0.0 and 1.0")
    
    # Validate tracking distance
    if args.tracking_distance <= 0:
        raise ValueError("Error: Tracking distance must be positive")
    
    # Validate model file exists
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Error: Model file '{args.model}' not found")
    
    # Parse input source (webcam index or file path)
    try:
        # Try to parse as integer (webcam device index)
        source = int(args.source)
        if source < 0:
            raise ValueError("Error: Webcam device index must be non-negative")
        return source
    except ValueError:
        # Not an integer, treat as file path
        if not os.path.exists(args.source):
            raise FileNotFoundError(f"Error: Video file '{args.source}' not found")
        return args.source


def main():
    """Main application entry point."""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        logger.info("=" * 60)
        logger.info("Person Detection and Counting System")
        logger.info("=" * 60)
        
        # Validate arguments
        input_source = validate_arguments(args)
        
        # Create configuration
        config = Config(
            model_path=args.model,
            confidence_threshold=args.confidence,
            tracking_distance=args.tracking_distance
        )
        
        logger.info(f"Configuration:")
        logger.info(f"  Input source: {input_source}")
        logger.info(f"  Output path: {args.output if args.output else 'None (display only)'}")
        logger.info(f"  Model: {config.model_path}")
        logger.info(f"  Confidence threshold: {config.confidence_threshold}")
        logger.info(f"  Tracking distance: {config.tracking_distance}")
        logger.info(f"  ROI file: {args.roi if args.roi else 'None (full frame)'}")
        
        # Initialize components
        logger.info("Initializing components...")
        
        detector = PersonDetector(
            model_path=config.model_path,
            confidence_threshold=config.confidence_threshold
        )
        
        tracker = PersonTracker(
            max_distance=config.tracking_distance
        )
        
        renderer = FrameRenderer(
            font_scale=config.font_scale,
            thickness=config.line_thickness,
            roi_color=config.roi_color,
            roi_thickness=config.roi_thickness,
            roi_alpha=config.roi_alpha
        )
        
        # Load ROI filter if specified
        roi_filter = None
        if args.roi:
            try:
                # We'll load with frame dimensions later in VideoProcessor
                # For now, create a placeholder that will be updated
                roi_filter = ROIFilter.load_from_file(args.roi)
                logger.info("ROI filter loaded successfully")
            except (FileNotFoundError, ValueError) as e:
                logger.error(f"Failed to load ROI configuration: {e}")
                logger.warning("Continuing with full frame detection")
                roi_filter = None
        
        processor = VideoProcessor(
            detector=detector,
            tracker=tracker,
            renderer=renderer,
            roi_filter=roi_filter
        )
        
        # Process video
        final_count = processor.process_video(
            input_source=input_source,
            output_path=args.output
        )
        
        # Display final statistics
        logger.info("=" * 60)
        logger.info(f"FINAL STATISTICS")
        logger.info(f"People in last frame: {final_count}")
        logger.info("=" * 60)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130
        
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
        
    except ValueError as e:
        logger.error(str(e))
        return 1
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
