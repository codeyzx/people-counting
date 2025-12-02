# Person Detection and Counting System

A clean, maintainable person detection and counting application using YOLOv8 with ONNX optimization and WebSocket integration. The system detects and counts unique people in video streams from webcam or video files, optimized for embedded devices like Orange Pi 5.

## Features

- **YOLOv8 Detection**: State-of-the-art person detection using YOLOv8
- **ONNX Runtime Support**: Optimized inference for ARM processors (Orange Pi 5)
- **WebSocket Integration**: Real-time event streaming to backend servers
- **Region of Interest (ROI)**: Define custom polygonal detection areas to focus on specific zones
- **Flexible Input**: Support for webcam devices or video files
- **Person Tracking**: Centroid-based tracking to maintain unique IDs across frames
- **Real-time Counting**: Displays current number of people in frame (count increases/decreases dynamically)
- **Real-time Visualization**: Live preview with bounding boxes, person IDs, and ROI overlay
- **Video Output**: Optional saving of processed video with visualizations
- **Automatic Fallback**: Falls back to PyTorch if ONNX model unavailable
- **Event Buffering**: Buffers events when WebSocket disconnected
- **Auto-Reconnection**: Exponential backoff reconnection for WebSocket
- **Clean Architecture**: Modular design with separation of concerns
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## Architecture

The system is organized into modular components:

```
src/
├── models.py              # Data models (Detection, TrackedPerson, DetectionEvent)
├── detector.py            # PersonDetector - YOLOv8 PyTorch detection
├── onnx_detector.py       # ONNXPersonDetector - ONNX Runtime detection
├── detector_factory.py    # Factory for creating detectors with fallback
├── model_converter.py     # Utility for converting PyTorch to ONNX
├── roi_filter.py          # ROIFilter - Region of Interest filtering
├── tracker.py             # PersonTracker - Tracking and counting
├── renderer.py            # FrameRenderer - Visualization
├── websocket_publisher.py # WebSocketPublisher - Event streaming
├── config_manager.py      # ConfigManager - Configuration loading
├── video_processor.py     # VideoProcessor - Pipeline orchestration
└── person_counter.py      # Main application entry point
```

### Module Responsibilities

- **Detection Module** (`detector.py`, `onnx_detector.py`): YOLOv8 detection with PyTorch or ONNX Runtime
- **Detector Factory** (`detector_factory.py`): Creates appropriate detector with automatic fallback
- **Model Converter** (`model_converter.py`): Converts YOLOv8 models to ONNX format
- **ROI Filter Module** (`roi_filter.py`): Filters detections based on Region of Interest polygon
- **Tracking Module** (`tracker.py`): Assigns unique IDs, tracks people across frames, maintains real-time count
- **Visualization Module** (`renderer.py`): Draws bounding boxes, IDs, ROI overlay, and current count
- **WebSocket Publisher** (`websocket_publisher.py`): Streams detection events to backend with buffering
- **Configuration Manager** (`config_manager.py`): Loads configuration from files or environment variables
- **Video Processor** (`video_processor.py`): Orchestrates the processing pipeline with WebSocket integration
- **Main Application** (`person_counter.py`): CLI interface and application entry point

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam (optional, for webcam input)
- For Orange Pi 5: ARM-compatible ONNX Runtime

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd PeopleCounting-ComputerVision
```

2. Create and activate virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure YOLOv8 model weights are available:
   - The `yolov8s.pt` file should be in the `models/` directory
   - If not present, it will be downloaded automatically on first run

### ONNX Model Setup

For optimized performance on Orange Pi 5 or other ARM devices:

1. **Convert PyTorch model to ONNX:**
```bash
python -m src.model_converter models/yolov8s.pt models/yolov8s.onnx
```

2. **Verify ONNX model:**
```bash
python -m src.model_converter --verify-only models/yolov8s.onnx
```

3. **Custom conversion options:**
```bash
python -m src.model_converter models/yolov8s.pt models/yolov8s.onnx \
  --input-size 640 640 \
  --opset 12 \
  --verbose
```

### Orange Pi 5 Deployment

For optimal performance on Orange Pi 5:

1. Install ONNX Runtime for ARM:
```bash
pip install onnxruntime
```

2. Use ONNX model with optimized settings:
```bash
python run.py \
  --onnx-model models/yolov8s.onnx \
  --source 0 \
  --confidence 0.6
```

3. Or use configuration file (recommended):
```bash
cp config.example.json config.json
# Edit config.json with your settings
python run.py --config config.json
```

## Usage

### Basic Usage

Run with default webcam (device 0):

**Option 1: Using wrapper script (recommended):**
```bash
python run.py
```

**Option 2: Using module:**
```bash
python -m src
```

**Option 3: Direct script:**
```bash
python src/person_counter.py
```

### Command-Line Options

```bash
python run.py [OPTIONS]
```

**Core Options:**

- `--source`: Video source (webcam index or video file path)
  - Default: `0` (default webcam)
  - Examples: `0`, `1`, `video.mp4`, `path/to/video.mp4`

- `--output`: Output video file path (optional)
  - Example: `output.avi`, `results/processed.avi`

- `--confidence`: Confidence threshold for detections (0.0 to 1.0)
  - Default: `0.5`

- `--tracking-distance`: Maximum distance for tracking across frames
  - Default: `50.0` pixels

- `--roi`: Path to ROI configuration JSON file (optional)
  - Example: `examples/roi_rectangle.json`

**Model Options:**

- `--model`: Path to PyTorch YOLOv8 model weights file
  - Default: `models/yolov8s.pt`

- `--onnx-model`: Path to ONNX model file (preferred for Orange Pi 5)
  - Example: `models/yolov8s.onnx`

**WebSocket Options:**

- `--ws-url`: WebSocket server URL
  - Example: `ws://localhost:8000/ws`

- `--no-websocket`: Disable WebSocket integration (standalone mode)

- `--source-id`: Unique identifier for this camera/source
  - Default: `camera_01`

**Configuration Options:**

- `--config`: Path to JSON configuration file
  - Example: `config.json`

- `--use-env`: Load configuration from environment variables

- `--help`: Show help message and exit

### Examples

**Basic Examples:**

```bash
# Use default webcam
python run.py

# Use specific webcam device
python run.py --source 1

# Process video file
python run.py --source test.mp4

# Process video and save output
python run.py --source test.mp4 --output result.avi

# Use custom confidence threshold
python run.py --source test.mp4 --confidence 0.7
```

**ONNX Examples:**

```bash
# Use ONNX model (optimized for Orange Pi 5)
python run.py --onnx-model models/yolov8s.onnx --source 0

# ONNX with fallback to PyTorch
python run.py --onnx-model models/yolov8s.onnx --model models/yolov8s.pt

# Standalone mode (no WebSocket)
python run.py --onnx-model models/yolov8s.onnx --no-websocket
```

**WebSocket Examples:**

```bash
# With WebSocket integration
python run.py --ws-url ws://localhost:8000/ws --source 0

# Custom source ID for multiple cameras
python run.py --ws-url ws://localhost:8000/ws --source-id camera_entrance

# WebSocket with ONNX
python run.py \
  --onnx-model models/yolov8s.onnx \
  --ws-url ws://backend.example.com/ws \
  --source-id camera_01
```

**Configuration File Examples:**

```bash
# Use configuration file
python run.py --config config.json --source 0

# Load from environment variables
python run.py --use-env --source 0
```

**ROI Examples:**

```bash
# Use rectangular ROI
python run.py --source test.mp4 --roi examples/roi_rectangle.json

# Use corridor-shaped ROI with normalized coordinates
python run.py --source test.mp4 --roi examples/roi_corridor.json --output result.avi

# Use doorway ROI for entrance monitoring
python run.py --source 0 --roi examples/roi_doorway.json
```

**Complete Example (Orange Pi 5 Production):**

```bash
python run.py \
  --onnx-model models/yolov8s.onnx \
  --model models/yolov8s.pt \
  --ws-url wss://backend.example.com/ws \
  --source-id camera_entrance \
  --source 0 \
  --confidence 0.6 \
  --roi examples/roi_doorway.json
```

### Controls

- **ESC key**: Stop processing and exit
- The application will also stop automatically when video file ends

## Output

### Console Output

The application provides detailed logging:
- Configuration settings
- Model loading status
- Processing progress
- Detection and tracking statistics
- Final count of unique people

### Video Output

When `--output` is specified, the processed video includes:
- Blue semi-transparent ROI polygon overlay (if ROI is enabled)
- Green bounding boxes around detected people
- Person IDs displayed near each bounding box
- Current count overlay in the top-left corner (real-time count)

### Statistics

At the end of processing, the application displays:
- Total frames processed
- Final count in last frame
- Processing time
- Average FPS

## Region of Interest (ROI) Configuration

### What is ROI?

ROI (Region of Interest) allows you to define specific areas within the video frame where detection and counting should be applied. This is useful for:
- Monitoring specific zones (doorways, corridors, designated areas)
- Ignoring irrelevant areas to improve accuracy
- Focusing on critical regions for security or analytics

### ROI Configuration File Format

ROI is defined using JSON files with the following structure:

```json
{
  "roi_points": [
    [x1, y1],
    [x2, y2],
    [x3, y3],
    [x4, y4]
  ],
  "coordinate_type": "absolute",
  "description": "Description of the ROI"
}
```

**Fields:**
- `roi_points`: List of (x, y) coordinates defining the polygon (minimum 3 points)
- `coordinate_type`: Either `"absolute"` (pixels) or `"normalized"` (0-1 range)
- `description`: Optional description of the ROI

### Coordinate Types

**Absolute Coordinates (pixels):**
```json
{
  "roi_points": [[200, 150], [1000, 150], [1000, 650], [200, 650]],
  "coordinate_type": "absolute"
}
```
- Coordinates are in pixels relative to frame dimensions
- Best for fixed camera setups with known resolution

**Normalized Coordinates (0-1 range):**
```json
{
  "roi_points": [[0.25, 0.2], [0.75, 0.2], [0.85, 0.9], [0.15, 0.9]],
  "coordinate_type": "normalized"
}
```
- Coordinates are normalized (0.0 to 1.0) relative to frame dimensions
- Resolution-independent, works with any video size
- Example: `[0.5, 0.5]` is always the center of the frame

### Example ROI Configurations

The `examples/` directory contains three pre-configured ROI files:

1. **roi_rectangle.json**: Rectangular region in center of frame (absolute coordinates)
2. **roi_corridor.json**: Trapezoid shape for corridor/hallway monitoring (normalized coordinates)
3. **roi_doorway.json**: Vertical rectangle for doorway/entrance monitoring (absolute coordinates)

### Creating Custom ROI

To create your own ROI configuration:

1. Determine the polygon points for your desired area
2. Choose coordinate type (absolute or normalized)
3. Create a JSON file with the structure shown above
4. Test with your video: `python run.py --source video.mp4 --roi your_roi.json`

**Tips:**
- Use at least 3 points (triangle) for a valid polygon
- Points should be ordered clockwise or counter-clockwise
- For complex shapes, use more points for better accuracy
- Normalized coordinates are recommended for flexibility across different resolutions

## Configuration

### Configuration File

Create a `config.json` file (see `config.example.json`):

```json
{
  "onnx": {
    "model_path": "models/yolov8s.onnx",
    "providers": ["CPUExecutionProvider"],
    "inter_op_num_threads": 4,
    "intra_op_num_threads": 4,
    "enable": true
  },
  "websocket": {
    "url": "ws://localhost:8000/ws",
    "reconnect_interval": 1.0,
    "max_reconnect_interval": 60.0,
    "buffer_size": 1000,
    "enable": true
  },
  "video": {
    "model_path": "models/yolov8s.pt",
    "confidence_threshold": 0.5,
    "tracking_distance": 50.0
  },
  "source_id": "camera_01"
}
```

### Environment Variables

Create a `.env` file (see `.env.example`):

```bash
# ONNX Configuration
ONNX_MODEL_PATH=models/yolov8s.onnx
ONNX_ENABLE=true
ONNX_PROVIDERS=CPUExecutionProvider
ONNX_THREADS=4

# WebSocket Configuration
WS_URL=ws://localhost:8000/ws
WS_ENABLE=true
WS_BUFFER_SIZE=1000

# Source Configuration
SOURCE_ID=camera_01

# Video/Detection Configuration
MODEL_PATH=models/yolov8s.pt
CONFIDENCE=0.5
TRACKING_DISTANCE=50.0
```

Then run with:
```bash
python run.py --use-env --source 0
```

## WebSocket Integration

### Event Format

The system sends detection events in JSON format:

```json
{
  "timestamp": "2024-12-02T10:30:45.123Z",
  "source_id": "camera_01",
  "frame_number": 1234,
  "event_type": "update",
  "current_count": 3,
  "tracked_persons": [
    {
      "person_id": 1,
      "bbox": [100, 150, 200, 400],
      "confidence": 0.92,
      "centroid": [150.0, 275.0]
    }
  ],
  "metadata": {
    "fps": 15.3,
    "inference_time_ms": 45.2
  }
}
```

### Event Types

- **update**: Count changed or periodic update
- **entry**: New person entered the frame
- **exit**: Person left the frame
- **lifecycle**: System started/stopped/error

### Lifecycle Events

```json
{
  "timestamp": "2024-12-02T10:30:00.000Z",
  "source_id": "camera_01",
  "event_type": "lifecycle",
  "lifecycle_event": "started",
  "metadata": {
    "model_type": "onnx",
    "model_path": "models/yolov8s.onnx"
  }
}
```

### Features

- **Auto-Reconnection**: Exponential backoff reconnection on disconnect
- **Event Buffering**: Buffers up to 1000 events when disconnected
- **FIFO Buffer**: Oldest events dropped when buffer full
- **Graceful Degradation**: Detection continues even if WebSocket fails

## Troubleshooting

### Webcam not working
- Check if webcam is connected and not used by another application
- Try different device indices: `--source 0`, `--source 1`, etc.
- On Linux, ensure you have proper permissions for `/dev/video*`

### Video file not found
- Verify the file path is correct
- Use absolute path if relative path doesn't work
- Ensure the video format is supported by OpenCV (MP4, AVI, etc.)

### Model file not found
- Ensure model files are in the `models/` directory
- For PyTorch: `models/yolov8s.pt`
- For ONNX: `models/yolov8s.onnx`
- Convert PyTorch to ONNX if needed (see ONNX Model Setup)

### ONNX Runtime issues
- **"onnxruntime not installed"**: Run `pip install onnxruntime`
- **"Failed to load ONNX model"**: Verify model file is valid with `--verify-only`
- **"Provider not available"**: Check available providers, system will fallback to CPU
- **Slow inference**: Ensure using correct providers for your hardware

### WebSocket connection issues
- **"Failed to connect"**: Verify WebSocket server is running and URL is correct
- **"websockets library not installed"**: Run `pip install websockets`
- **Connection keeps dropping**: Check network stability, system will auto-reconnect
- **Events not received**: Check buffer size, may be full and dropping old events
- Use `--no-websocket` flag to run in standalone mode for testing

### Low FPS / Slow processing
- Use ONNX model instead of PyTorch (significant speedup on ARM)
- Use a smaller model (yolov8n.pt/onnx instead of yolov8s.pt/onnx)
- Increase confidence threshold to reduce detections
- Reduce ONNX thread count if CPU is overloaded
- On Orange Pi 5: Use 4-6 threads for optimal performance

### Memory issues
- Close other applications
- Use a smaller model
- Reduce video resolution
- Reduce WebSocket buffer size in configuration

### ROI not working
- Verify the JSON file path is correct
- Check JSON syntax is valid (use a JSON validator)
- Ensure `roi_points` has at least 3 coordinate pairs
- For absolute coordinates, ensure points are within frame dimensions
- For normalized coordinates, ensure values are between 0.0 and 1.0
- Check the log output for specific error messages

### Configuration issues
- **"Invalid JSON"**: Check configuration file syntax
- **"Invalid confidence threshold"**: Must be between 0.0 and 1.0
- **"Invalid WebSocket URL"**: Must start with ws:// or wss://
- System will use default values for invalid configurations

## Dependencies

Core dependencies:
- **ultralytics**: YOLOv8 implementation
- **opencv-python**: Video I/O and visualization
- **numpy**: Array operations
- **onnx**: ONNX model format support
- **onnxruntime**: Optimized inference engine
- **websockets**: WebSocket client for real-time communication
- **python-dotenv**: Environment variable management

See `requirements.txt` for complete list with versions.

## Performance

- **Model**: YOLOv8s (small) - balance between speed and accuracy
- **ONNX Optimization**: 2-3x faster inference on ARM processors
- **Orange Pi 5**: 10-15 FPS with ONNX (vs 3-5 FPS with PyTorch)
- **Processing**: Real-time capable on modern hardware
- **Tracking**: Centroid-based tracking with configurable distance threshold
- **WebSocket**: Minimal overhead with event buffering

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Acknowledgments

- YOLOv8 by Ultralytics
- OpenCV community

## Legacy Files

Previous implementation files are documented in `LEGACY_FILES.md`. The new implementation provides:
- Cleaner architecture with separation of concerns
- Better error handling and logging
- Simplified counting (no bidirectional tracking)
- More flexible input/output options
