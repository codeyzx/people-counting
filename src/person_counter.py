"""
Person Detection and Counting Application.

Main entry point for the person detection and counting system using YOLOv8.
Supports flexible input from webcam or video files with optional video output.
Enhanced with ONNX support and WebSocket integration.
"""

import argparse
import logging
import sys
import os
import asyncio
from pathlib import Path
from .models import Config, StreamingConfig
from .detector import PersonDetector
from .tracker import PersonTracker
from .renderer import FrameRenderer
from .video_processor import VideoProcessor
from .roi_filter import ROIFilter
from .config_manager import ConfigManager, SystemConfig
from .detector_factory import create_detector, validate_model_file, log_detector_info
from .websocket_publisher import WebSocketPublisher
from .frame_encoder import FrameEncoder


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
        description='Person Detection and Counting System with ONNX and WebSocket support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default webcam with ONNX model
  python person_counter.py --onnx-model models/yolov8s.onnx
  
  # Use PyTorch model (fallback)
  python person_counter.py --model models/yolov8s.pt
  
  # With WebSocket integration
  python person_counter.py --ws-url ws://localhost:8000/ws
  
  # Standalone mode (no WebSocket)
  python person_counter.py --no-websocket
  
  # Load from config file
  python person_counter.py --config config.json
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
        help='Output video file path (optional)'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=None,
        help='Confidence threshold for detections (0.0 to 1.0)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to PyTorch YOLOv8 model weights file'
    )
    
    parser.add_argument(
        '--onnx-model',
        type=str,
        default=None,
        help='Path to ONNX model file (preferred for Orange Pi 5)'
    )
    
    parser.add_argument(
        '--tracking-distance',
        type=float,
        default=None,
        help='Maximum distance for tracking people across frames'
    )
    
    parser.add_argument(
        '--roi',
        type=str,
        default=None,
        help='Path to ROI configuration JSON file (optional)'
    )
    
    parser.add_argument(
        '--ws-url',
        type=str,
        default=None,
        help='WebSocket server URL (e.g., ws://localhost:8000/ws)'
    )
    
    parser.add_argument(
        '--no-websocket',
        action='store_true',
        help='Disable WebSocket integration (standalone mode)'
    )
    
    parser.add_argument(
        '--source-id',
        type=str,
        default=None,
        help='Unique identifier for this camera/source'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to JSON configuration file'
    )
    
    parser.add_argument(
        '--use-env',
        action='store_true',
        help='Load configuration from environment variables'
    )
    
    parser.add_argument(
        '--enable-streaming',
        action='store_true',
        help='Enable video frame streaming to dashboard'
    )
    
    parser.add_argument(
        '--max-frame-rate',
        type=float,
        default=10.0,
        help='Maximum frame rate for streaming (FPS, default: 10.0)'
    )
    
    parser.add_argument(
        '--jpeg-quality',
        type=int,
        default=85,
        help='JPEG quality for frame encoding (1-100, default: 85)'
    )
    
    parser.add_argument(
        '--max-frame-width',
        type=int,
        default=1280,
        help='Maximum frame width for streaming (default: 1280)'
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate command-line arguments.
    
    Args:
        args: Parsed arguments namespace
        
    Raises:
        ValueError: If arguments are invalid
    """
    # Validate confidence threshold (only if provided)
    if args.confidence is not None:
        if not 0.0 <= args.confidence <= 1.0:
            raise ValueError("Error: Confidence threshold must be between 0.0 and 1.0")
    
    # Validate tracking distance (only if provided)
    if args.tracking_distance is not None:
        if args.tracking_distance <= 0:
            raise ValueError("Error: Tracking distance must be positive")
    
    # Validate model file exists (only if provided)
    if args.model is not None:
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


async def async_main():
    """Async main application entry point."""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    websocket_publisher = None
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        logger.info("=" * 60)
        logger.info("Person Detection and Counting System")
        logger.info("With ONNX and WebSocket Support")
        logger.info("=" * 60)
        
        # Load configuration
        if args.config:
            logger.info(f"Loading configuration from file: {args.config}")
            system_config = ConfigManager.load_from_file(args.config)
        elif args.use_env:
            logger.info("Loading configuration from environment variables")
            system_config = ConfigManager.load_from_env()
        else:
            # Create default config and override with CLI args
            system_config = SystemConfig()
            
            # Override with CLI arguments
            if args.onnx_model:
                system_config.onnx.model_path = args.onnx_model
            if args.model:
                system_config.video.model_path = args.model
            if args.confidence is not None:
                system_config.video.confidence_threshold = args.confidence
            if args.tracking_distance is not None:
                system_config.video.tracking_distance = args.tracking_distance
            if args.ws_url:
                system_config.websocket.url = args.ws_url
            if args.no_websocket:
                system_config.websocket.enable = False
            if args.source_id:
                system_config.source_id = args.source_id
        
        # Validate input source
        input_source = validate_arguments(args)
        
        # Log configuration
        logger.info(f"Configuration:")
        logger.info(f"  Input source: {input_source}")
        logger.info(f"  Output path: {args.output if args.output else 'None (display only)'}")
        logger.info(f"  Source ID: {system_config.source_id}")
        logger.info(f"  ONNX model: {system_config.onnx.model_path}")
        logger.info(f"  ONNX enabled: {system_config.onnx.enable}")
        logger.info(f"  PyTorch model: {system_config.video.model_path}")
        logger.info(f"  Confidence threshold: {system_config.video.confidence_threshold}")
        logger.info(f"  Tracking distance: {system_config.video.tracking_distance}")
        logger.info(f"  WebSocket enabled: {system_config.websocket.enable}")
        if system_config.websocket.enable:
            logger.info(f"  WebSocket URL: {system_config.websocket.url}")
        logger.info(f"  ROI file: {args.roi if args.roi else 'None (full frame)'}")
        
        # Initialize components
        logger.info("Initializing components...")
        
        # Create detector with fallback
        detector = create_detector(
            onnx_model_path=system_config.onnx.model_path if system_config.onnx.enable else None,
            pytorch_model_path=system_config.video.model_path,
            confidence_threshold=system_config.video.confidence_threshold,
            use_onnx=system_config.onnx.enable,
            onnx_providers=system_config.onnx.providers,
            inter_op_threads=system_config.onnx.inter_op_num_threads,
            intra_op_threads=system_config.onnx.intra_op_num_threads
        )
        
        # Log detector info
        log_detector_info(detector)
        
        # Create tracker
        tracker = PersonTracker(
            max_distance=system_config.video.tracking_distance
        )
        
        # Create renderer
        renderer = FrameRenderer(
            font_scale=system_config.video.font_scale,
            thickness=system_config.video.line_thickness,
            roi_color=system_config.video.roi_color,
            roi_thickness=system_config.video.roi_thickness,
            roi_alpha=system_config.video.roi_alpha
        )
        
        # Load ROI filter if specified
        roi_filter = None
        if args.roi:
            try:
                roi_filter = ROIFilter.load_from_file(args.roi)
                logger.info("ROI filter loaded successfully")
            except (FileNotFoundError, ValueError) as e:
                logger.error(f"Failed to load ROI configuration: {e}")
                logger.warning("Continuing with full frame detection")
                roi_filter = None
        
        # Create streaming config
        streaming_config = StreamingConfig()
        if args.enable_streaming:
            streaming_config.enable_streaming = True
        if args.max_frame_rate:
            streaming_config.max_frame_rate = args.max_frame_rate
        if args.jpeg_quality:
            streaming_config.jpeg_quality = args.jpeg_quality
        if args.max_frame_width:
            streaming_config.max_frame_width = args.max_frame_width
        
        # Initialize WebSocket publisher if enabled
        if system_config.websocket.enable:
            logger.info("Initializing WebSocket publisher...")
            
            websocket_publisher = WebSocketPublisher(
                url=system_config.websocket.url,
                buffer_size=system_config.websocket.buffer_size,
                frame_buffer_size=streaming_config.frame_buffer_size,
                reconnect_interval=system_config.websocket.reconnect_interval,
                max_reconnect_interval=system_config.websocket.max_reconnect_interval
            )
            
            # Try to connect
            connected = await websocket_publisher.connect()
            if connected:
                logger.info("✓ WebSocket connected")
            else:
                logger.warning("WebSocket connection failed, will retry in background")
        
        # Initialize frame encoder if streaming is enabled
        frame_encoder = None
        if system_config.websocket.enable and args.enable_streaming:
            logger.info("Initializing frame encoder for streaming...")
            
            frame_encoder = FrameEncoder(
                quality=streaming_config.jpeg_quality,
                max_width=streaming_config.max_frame_width,
                max_height=streaming_config.max_frame_height
            )
            logger.info(f"Frame streaming enabled: {streaming_config.max_frame_rate} FPS, quality={streaming_config.jpeg_quality}")
        
        # Create video processor
        processor = VideoProcessor(
            detector=detector,
            tracker=tracker,
            renderer=renderer,
            roi_filter=roi_filter,
            websocket_publisher=websocket_publisher,
            frame_encoder=frame_encoder,
            source_id=system_config.source_id,
            max_frame_rate=streaming_config.max_frame_rate if frame_encoder else None
        )
        
        # Process video
        logger.info("Starting video processing...")
        final_count = await processor.process_video_async(
            input_source=input_source,
            output_path=args.output
        )
        
        # Display final statistics
        logger.info("=" * 60)
        logger.info(f"FINAL STATISTICS")
        logger.info(f"People in last frame: {final_count}")
        if websocket_publisher:
            logger.info(f"Buffered events: {websocket_publisher.get_buffer_size()}")
            logger.info(f"Buffered frames: {websocket_publisher.get_frame_buffer_size()}")
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
        
    finally:
        # Cleanup WebSocket connection
        if websocket_publisher:
            logger.info("Closing WebSocket connection...")
            try:
                await websocket_publisher.disconnect()
            except Exception as e:
                logger.warning(f"Error during WebSocket cleanup: {e}")


def main():
    """Main entry point wrapper."""
    return asyncio.run(async_main())


if __name__ == "__main__":
    sys.exit(main())
