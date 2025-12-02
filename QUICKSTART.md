# Quick Start Guide

## For Orange Pi 5 Deployment

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Convert Model to ONNX

```bash
# Convert YOLOv8 model to ONNX format
python convert_model.py models/yolov8s.pt models/yolov8s.onnx

# Verify conversion
python convert_model.py --verify-only models/yolov8s.onnx
```

### 3. Configure System

**Option A: Using Configuration File**

```bash
# Copy example config
cp config.example.json config.json

# Edit config.json with your settings
nano config.json

# Run with config
python run.py --config config.json --source 0
```

**Option B: Using Environment Variables**

```bash
# Copy example env file
cp .env.example .env

# Edit .env with your settings
nano .env

# Run with env vars
python run.py --use-env --source 0
```

**Option C: Using Command Line**

```bash
# Run with ONNX and WebSocket
python run.py \
  --onnx-model models/yolov8s.onnx \
  --ws-url ws://your-backend:8000/ws \
  --source-id camera_01 \
  --source 0
```

### 4. Test Without WebSocket (Standalone Mode)

```bash
# Test ONNX model without WebSocket
python run.py \
  --onnx-model models/yolov8s.onnx \
  --no-websocket \
  --source 0
```

### 5. Production Deployment

```bash
# Full production setup
python run.py \
  --onnx-model models/yolov8s.onnx \
  --model models/yolov8s.pt \
  --ws-url wss://backend.example.com/ws \
  --source-id camera_entrance \
  --source 0 \
  --confidence 0.6 \
  --roi examples/roi_doorway.json
```

## Common Use Cases

### Webcam Monitoring with WebSocket

```bash
python run.py \
  --onnx-model models/yolov8s.onnx \
  --ws-url ws://localhost:8000/ws \
  --source 0
```

### Video File Processing

```bash
python run.py \
  --onnx-model models/yolov8s.onnx \
  --source video.mp4 \
  --output result.avi \
  --no-websocket
```

### Multiple Camera Setup

```bash
# Camera 1
python run.py \
  --config config_camera1.json \
  --source 0 \
  --source-id camera_entrance

# Camera 2 (in another terminal)
python run.py \
  --config config_camera2.json \
  --source 1 \
  --source-id camera_exit
```

## Troubleshooting

### Model Conversion Fails

```bash
# Check if ultralytics is installed
pip install ultralytics

# Try with verbose output
python convert_model.py models/yolov8s.pt models/yolov8s.onnx --verbose
```

### ONNX Runtime Not Found

```bash
# Install ONNX Runtime
pip install onnxruntime

# For GPU support (if available)
pip install onnxruntime-gpu
```

### WebSocket Connection Issues

```bash
# Test without WebSocket first
python run.py --onnx-model models/yolov8s.onnx --no-websocket --source 0

# Check if websockets is installed
pip install websockets

# Verify WebSocket server is running
# curl -i -N -H "Connection: Upgrade" -H "Upgrade: websocket" http://localhost:8000/ws
```

### Low FPS on Orange Pi 5

```bash
# Use optimized settings
python run.py \
  --onnx-model models/yolov8s.onnx \
  --confidence 0.6 \
  --source 0

# Or edit config.json:
# "onnx": {
#   "inter_op_num_threads": 6,
#   "intra_op_num_threads": 6
# }
```

## Performance Tips

1. **Use ONNX model** - 2-3x faster than PyTorch on ARM
2. **Adjust confidence threshold** - Higher = faster (fewer detections)
3. **Optimize thread count** - 4-6 threads optimal for Orange Pi 5
4. **Use ROI** - Reduces processing area
5. **Disable video output** - Don't use `--output` for production

## Next Steps

- Set up backend WebSocket server to receive events
- Configure ROI for specific monitoring areas
- Set up systemd service for auto-start
- Configure multiple cameras with different source IDs
- Monitor logs for performance metrics
