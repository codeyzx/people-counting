# Person Detection and Counting System

A clean, maintainable person detection and counting application using YOLOv8. The system detects and counts unique people in video streams from webcam or video files, with optional video output.

## Features

- **YOLOv8 Detection**: State-of-the-art person detection using YOLOv8
- **Region of Interest (ROI)**: Define custom polygonal detection areas to focus on specific zones
- **Flexible Input**: Support for webcam devices or video files
- **Person Tracking**: Centroid-based tracking to maintain unique IDs across frames
- **Real-time Counting**: Displays current number of people in frame (count increases/decreases dynamically)
- **Real-time Visualization**: Live preview with bounding boxes, person IDs, and ROI overlay
- **Video Output**: Optional saving of processed video with visualizations
- **Clean Architecture**: Modular design with separation of concerns
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## Architecture

The system is organized into four main modules:

```
src/
├── models.py           # Data models (Detection, TrackedPerson, Config, ROIConfig)
├── detector.py         # PersonDetector - YOLOv8 detection
├── roi_filter.py       # ROIFilter - Region of Interest filtering
├── tracker.py          # PersonTracker - Tracking and counting
├── renderer.py         # FrameRenderer - Visualization
├── video_processor.py  # VideoProcessor - Pipeline orchestration
└── person_counter.py   # Main application entry point
```

### Module Responsibilities

- **Detection Module** (`detector.py`): Loads YOLOv8 model, detects people, filters by confidence
- **ROI Filter Module** (`roi_filter.py`): Filters detections based on Region of Interest polygon
- **Tracking Module** (`tracker.py`): Assigns unique IDs, tracks people across frames, maintains real-time count
- **Visualization Module** (`renderer.py`): Draws bounding boxes, IDs, ROI overlay, and current count
- **Video Processor** (`video_processor.py`): Orchestrates the processing pipeline
- **Main Application** (`person_counter.py`): CLI interface and application entry point

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam (optional, for webcam input)

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
   - The `yolov8s.pt` file should be in the project root
   - If not present, it will be downloaded automatically on first run

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

Or:
```bash
python -m src [OPTIONS]
```

**Options:**

- `--source`: Video source (webcam index or video file path)
  - Default: `0` (default webcam)
  - Examples: `0`, `1`, `video.mp4`, `path/to/video.mp4`

- `--output`: Output video file path (optional)
  - If not specified, video will not be saved
  - Example: `output.avi`, `results/processed.avi`

- `--confidence`: Confidence threshold for detections (0.0 to 1.0)
  - Default: `0.5`
  - Higher values = fewer but more confident detections

- `--model`: Path to YOLOv8 model weights file
  - Default: `yolov8s.pt`

- `--tracking-distance`: Maximum distance for tracking across frames
  - Default: `50.0` pixels
  - Lower values = stricter tracking

- `--roi`: Path to ROI configuration JSON file (optional)
  - If not specified, full frame detection is used
  - Example: `examples/roi_rectangle.json`

- `--help`: Show help message and exit

### Examples

**Use default webcam:**
```bash
python run.py
```

**Use specific webcam device:**
```bash
python run.py --source 1
```

**Process video file:**
```bash
python run.py --source test.mp4
```

**Process video and save output:**
```bash
python run.py --source test.mp4 --output result.avi
```

**Use custom confidence threshold:**
```bash
python run.py --source test.mp4 --confidence 0.7
```

**Combine multiple options:**
```bash
python run.py --source test.mp4 --output result.avi --confidence 0.6 --tracking-distance 40
```

**Use ROI (Region of Interest):**
```bash
# Use rectangular ROI
python run.py --source test.mp4 --roi examples/roi_rectangle.json

# Use corridor-shaped ROI with normalized coordinates
python run.py --source test.mp4 --roi examples/roi_corridor.json --output result.avi

# Use doorway ROI for entrance monitoring
python run.py --source 0 --roi examples/roi_doorway.json
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

Default configuration can be modified in `src/models.py`:

```python
@dataclass
class Config:
    model_path: str = "models/yolov8s.pt"
    confidence_threshold: float = 0.5
    tracking_distance: float = 50.0
    font_scale: float = 0.6
    line_thickness: int = 2
    window_name: str = "Person Detection & Counting"
    roi_file: Optional[str] = None
    roi_color: Tuple[int, int, int] = (255, 0, 0)  # Blue (BGR)
    roi_thickness: int = 2
    roi_alpha: float = 0.3  # Transparency
```

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
- Ensure `yolov8s.pt` is in the project root
- Or specify the correct path with `--model` option
- The model will be downloaded automatically on first run if using ultralytics

### Low FPS / Slow processing
- Use a smaller model (yolov8n.pt instead of yolov8s.pt)
- Increase confidence threshold to reduce detections
- Process every N-th frame instead of every frame

### Memory issues
- Close other applications
- Use a smaller model
- Reduce video resolution

### ROI not working
- Verify the JSON file path is correct
- Check JSON syntax is valid (use a JSON validator)
- Ensure `roi_points` has at least 3 coordinate pairs
- For absolute coordinates, ensure points are within frame dimensions
- For normalized coordinates, ensure values are between 0.0 and 1.0
- Check the log output for specific error messages

### ROI configuration errors
- **"ROI polygon must have at least 3 points"**: Add more coordinate points to your polygon
- **"Invalid JSON"**: Check for syntax errors (missing commas, brackets, quotes)
- **"ROI configuration file not found"**: Verify the file path is correct
- If ROI loading fails, the system will automatically fall back to full frame detection

## Dependencies

- **ultralytics**: YOLOv8 implementation
- **opencv-python**: Video I/O and visualization
- **numpy**: Array operations

See `requirements.txt` for complete list with versions.

## Performance

- **Model**: YOLOv8s (small) - balance between speed and accuracy
- **Processing**: Real-time capable on modern hardware
- **Tracking**: Centroid-based tracking with configurable distance threshold

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
