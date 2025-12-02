# Person Detection & Counting System with Live Streaming

A production-ready real-time person detection and counting system with live video streaming dashboard. Built with YOLOv8, WebSocket, and React for monitoring multiple camera feeds simultaneously.

## 🎯 Key Features

- **Real-time Person Detection** - YOLOv8-based detection with centroid tracking
- **Live Video Streaming** - Stream detection results to web dashboard via WebSocket
- **Multi-Camera Support** - Monitor multiple camera feeds simultaneously
- **ROI (Region of Interest)** - Define custom detection zones
- **ONNX Optimization** - Optimized for ARM processors (Orange Pi 5)
- **Web Dashboard** - React-based real-time monitoring interface
- **Smart Logging** - Conditional logging based on client connections
- **Auto-Reconnection** - Exponential backoff for WebSocket connections

## 🏗️ Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│  Detection      │      │   WebSocket      │      │   Dashboard     │
│  Device(s)      │─────▶│   Server         │─────▶│   (React)       │
│  (Python)       │      │   (Port 8000)    │      │   (Port 5173)   │
└─────────────────┘      └──────────────────┘      └─────────────────┘
     │                            │                          │
     │ - YOLOv8 Detection        │ - Message Routing        │ - Live Video
     │ - Frame Encoding          │ - Client Management      │ - Person Count
     │ - ROI Filtering           │ - Broadcasting           │ - Device Status
     └───────────────────────────┴──────────────────────────┘
```

### System Components

**Backend (Python):**
- `backend_server.py` - WebSocket server (routes messages)
- `src/person_counter.py` - Main detection application
- `src/video_processor.py` - Video processing pipeline
- `src/frame_encoder.py` - JPEG encoding for streaming
- `src/websocket_publisher.py` - WebSocket client for devices
- `src/detector.py` / `src/onnx_detector.py` - YOLOv8 detection
- `src/tracker.py` - Centroid-based person tracking
- `src/roi_filter.py` - Region of Interest filtering
- `src/renderer.py` - Visualization rendering

**Frontend (React + TypeScript):**
- `frontend/src/App.tsx` - Main application
- `frontend/src/hooks/useWebSocket.ts` - WebSocket connection management
- `frontend/src/hooks/useDeviceState.tsx` - Device state management
- `frontend/src/components/LivePreview.tsx` - Video stream display
- `frontend/src/components/DeviceCard.tsx` - Device monitoring card
- `frontend/src/components/DeviceGrid.tsx` - Grid layout for devices

## 📋 Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **Node.js**: 16.x or higher
- **npm**: 8.x or higher
- **OS**: Windows, Linux, or macOS
- **RAM**: 2GB minimum, 4GB recommended
- **CPU**: 4+ cores recommended for real-time processing

### Hardware (Optional)

- **Webcam**: For live camera input
- **GPU**: CUDA-compatible GPU for faster inference (optional)
- **Orange Pi 5**: Supported with ONNX optimization

## 🚀 Installation & Setup

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd PeopleCounting-ComputerVision
```

### Step 2: Backend Setup

#### 2.1 Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

#### 2.2 Install Python Dependencies

```bash
pip install -r requirements.txt
```

#### 2.3 Verify Model Files

Ensure YOLOv8 model exists:
```bash
# Check if model exists
ls models/yolov8s.pt

# If not, it will be downloaded automatically on first run
```

#### 2.4 (Optional) Convert to ONNX for Optimization

```bash
# Convert PyTorch model to ONNX
python -m src.model_converter models/yolov8s.pt models/yolov8s.onnx

# Verify ONNX model
python -m src.model_converter --verify-only models/yolov8s.onnx
```

### Step 3: Frontend Setup

#### 3.1 Navigate to Frontend Directory

```bash
cd frontend
```

#### 3.2 Install Node Dependencies

```bash
npm install
```

#### 3.3 Configure Environment (Optional)

```bash
# Copy environment template
cp .env.example .env

# Edit .env if needed (default values work for local development)
```

#### 3.4 Return to Root Directory

```bash
cd ..
```

## 🎮 Running the System

### Complete System (3 Terminals)

#### Terminal 1: Start WebSocket Server

```bash
python backend_server.py
```

**Expected Output:**
```
INFO - Starting WebSocket server on 0.0.0.0:8000
INFO - Dashboard clients connect to: ws://0.0.0.0:8000/ws
INFO - Detection devices connect to: ws://0.0.0.0:8000/device
INFO - WebSocket server started successfully
```

#### Terminal 2: Start Frontend Dashboard

```bash
cd frontend
npm run dev
```

**Expected Output:**
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

Open browser: **http://localhost:5173**

#### Terminal 3: Start Detection

**With Webcam:**
```bash
python run.py --source 0 --enable-streaming --max-frame-rate 10
```

**With Video File:**
```bash
python run.py --source assets/q1.mp4 --enable-streaming --max-frame-rate 10
```

**With ROI:**
```bash
python run.py --source assets/q1.mp4 --enable-streaming --roi examples/roi_corridor.json
```

### Standalone Mode (No Dashboard)

Run detection without WebSocket/Dashboard:

```bash
python run.py --source 0 --no-websocket
```

This will show OpenCV window with detections (no streaming).

## 📖 Usage Guide

### Basic Commands

#### Webcam Input

```bash
# Default webcam (device 0)
python run.py --source 0 --enable-streaming

# Specific webcam device
python run.py --source 1 --enable-streaming

# With custom settings
python run.py --source 0 --enable-streaming --max-frame-rate 15 --jpeg-quality 90
```

#### Video File Input

```bash
# Basic video processing
python run.py --source assets/q1.mp4 --enable-streaming

# With output file
python run.py --source assets/q1.mp4 --enable-streaming --output result.avi

# With custom source ID
python run.py --source assets/q1.mp4 --enable-streaming --source-id "Video_Test_1"
```

#### ROI (Region of Interest)

```bash
# Corridor ROI (trapezoid, normalized coordinates)
python run.py --source assets/q1.mp4 --enable-streaming --roi examples/roi_corridor.json

# Doorway ROI (rectangle, absolute coordinates)
python run.py --source 0 --enable-streaming --roi examples/roi_doorway.json

# Custom ROI
python run.py --source 0 --enable-streaming --roi my_custom_roi.json
```

#### Multiple Cameras

Run multiple detection instances with different source IDs:

```bash
# Terminal 3a - Camera 1
python run.py --source 0 --enable-streaming --source-id "Camera_Entrance"

# Terminal 3b - Camera 2
python run.py --source 1 --enable-streaming --source-id "Camera_Exit"

# Terminal 3c - Video File
python run.py --source assets/q1.mp4 --enable-streaming --source-id "Video_Archive"
```

Dashboard will show all 3 devices simultaneously!

### Command-Line Options

#### Core Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--source` | Video source (webcam index or file path) | `0` | `0`, `1`, `video.mp4` |
| `--output` | Output video file path | None | `result.avi` |
| `--confidence` | Detection confidence threshold (0.0-1.0) | `0.5` | `0.7` |
| `--tracking-distance` | Max tracking distance (pixels) | `50.0` | `40.0` |
| `--roi` | ROI configuration JSON file | None | `examples/roi_corridor.json` |

#### Model Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--model` | PyTorch YOLOv8 model path | `models/yolov8s.pt` | `models/yolov8n.pt` |
| `--onnx-model` | ONNX model path (faster) | None | `models/yolov8s.onnx` |

#### Streaming Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--enable-streaming` | Enable video frame streaming | `False` | (flag) |
| `--max-frame-rate` | Max streaming FPS | `10.0` | `15.0` |
| `--jpeg-quality` | JPEG quality (1-100) | `85` | `90` |
| `--max-frame-width` | Max frame width for streaming | `1280` | `1920` |

#### WebSocket Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--ws-url` | WebSocket server URL | `ws://localhost:8000/device` | `ws://192.168.1.100:8000/device` |
| `--no-websocket` | Disable WebSocket (standalone) | `False` | (flag) |
| `--source-id` | Unique camera identifier | `camera_01` | `Camera_Entrance` |

#### Configuration Options

| Option | Description | Example |
|--------|-------------|---------|
| `--config` | Load from JSON config file | `config.json` |
| `--use-env` | Load from environment variables | (flag) |

### Advanced Examples

#### Production Setup with ONNX

```bash
python run.py \
  --onnx-model models/yolov8s.onnx \
  --source 0 \
  --enable-streaming \
  --roi examples/roi_doorway.json \
  --source-id "Entrance_Camera" \
  --confidence 0.6 \
  --max-frame-rate 10 \
  --jpeg-quality 85
```

#### High-Quality Streaming

```bash
python run.py \
  --source 0 \
  --enable-streaming \
  --max-frame-rate 15 \
  --jpeg-quality 95 \
  --max-frame-width 1920
```

#### Low-Bandwidth Streaming

```bash
python run.py \
  --source 0 \
  --enable-streaming \
  --max-frame-rate 5 \
  --jpeg-quality 70 \
  --max-frame-width 640
```

#### Video Analysis with Output

```bash
python run.py \
  --source assets/q1.mp4 \
  --enable-streaming \
  --roi examples/roi_corridor.json \
  --output analysis_result.avi \
  --confidence 0.7
```

## 🎯 ROI Configuration

### What is ROI?

ROI (Region of Interest) allows you to define specific areas for detection. Only people inside the ROI will be counted.

### ROI File Format

```json
{
  "roi_points": [
    [x1, y1],
    [x2, y2],
    [x3, y3],
    [x4, y4]
  ],
  "coordinate_type": "normalized",
  "description": "Description of ROI"
}
```

### Coordinate Types

**Normalized (0-1 range) - Recommended:**
```json
{
  "roi_points": [
    [0.25, 0.2],   // Top-left (25% from left, 20% from top)
    [0.75, 0.2],   // Top-right
    [0.85, 0.9],   // Bottom-right
    [0.15, 0.9]    // Bottom-left
  ],
  "coordinate_type": "normalized"
}
```

**Absolute (pixels):**
```json
{
  "roi_points": [
    [200, 150],    // Top-left (200px, 150px)
    [1000, 150],   // Top-right
    [1000, 650],   // Bottom-right
    [200, 650]     // Bottom-left
  ],
  "coordinate_type": "absolute"
}
```

### Pre-made ROI Examples

| File | Shape | Coordinates | Use Case |
|------|-------|-------------|----------|
| `roi_corridor.json` | Trapezoid | Normalized | Hallways, corridors |
| `roi_doorway.json` | Rectangle | Absolute | Doorways, entrances |
| `roi_rectangle.json` | Rectangle | Absolute | Center area monitoring |

### Creating Custom ROI

1. **Determine coordinates** (use video player or OpenCV script)
2. **Create JSON file** with 3+ points (polygon)
3. **Test with video:**
   ```bash
   python run.py --source test.mp4 --enable-streaming --roi my_roi.json
   ```
4. **Verify in dashboard** (ROI shown as blue polygon)

**See `ROI_GUIDE.md` for detailed instructions.**

## 🔧 Configuration Files

### Backend Configuration (`config.json`)

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
    "url": "ws://localhost:8000/device",
    "reconnect_interval": 1.0,
    "max_reconnect_interval": 60.0,
    "buffer_size": 1000,
    "frame_buffer_size": 10,
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

### Frontend Configuration (`frontend/.env`)

```bash
# WebSocket Server URL
VITE_WS_URL=ws://localhost:8000/ws

# Offline detection threshold (ms)
VITE_OFFLINE_THRESHOLD=30000

# Device status check interval (ms)
VITE_CHECK_INTERVAL=5000
```

### Environment Variables (`.env`)

```bash
# ONNX Configuration
ONNX_MODEL_PATH=models/yolov8s.onnx
ONNX_ENABLE=true
ONNX_PROVIDERS=CPUExecutionProvider
ONNX_THREADS=4

# WebSocket Configuration
WS_URL=ws://localhost:8000/device
WS_ENABLE=true
WS_BUFFER_SIZE=1000

# Source Configuration
SOURCE_ID=camera_01

# Video/Detection Configuration
MODEL_PATH=models/yolov8s.pt
CONFIDENCE=0.5
TRACKING_DISTANCE=50.0
```

## 📡 WebSocket Protocol

### Message Types

#### 1. Detection Message

```json
{
  "type": "detection",
  "timestamp": "2024-12-02T10:30:45.123Z",
  "source_id": "camera_01",
  "frame_number": 1234,
  "event_type": "update",
  "current_count": 5,
  "tracked_persons": [
    {
      "person_id": 1,
      "bbox": [100, 200, 300, 500],
      "confidence": 0.95,
      "centroid": [200, 350]
    }
  ],
  "metadata": {
    "fps": 28.5,
    "inference_time_ms": 35.2
  }
}
```

#### 2. Frame Message

```json
{
  "type": "frame",
  "timestamp": "2024-12-02T10:30:45.123Z",
  "source_id": "camera_01",
  "frame_number": 1234,
  "frame": "/9j/4AAQSkZJRgABAQAAAQABAAD...",  // Base64 JPEG
  "metadata": {
    "fps": 10.0,
    "resolution": "1280x720",
    "quality": 85
  }
}
```

### Connection Endpoints

- **Dashboard**: `ws://localhost:8000/ws`
- **Detection Devices**: `ws://localhost:8000/device`

### Event Types

| Event Type | Description | Trigger |
|------------|-------------|---------|
| `update` | Count changed or periodic update | Count change or interval |
| `entry` | New person entered frame | New person detected |
| `exit` | Person left frame | Person no longer detected |
| `lifecycle` | System event | Start/stop/error |

## 🐛 Troubleshooting

### Backend Issues

#### WebSocket Server Won't Start

**Symptom:** "Address already in use" error

**Solution:**
```bash
# Find process using port 8000
netstat -ano | findstr "8000"

# Kill the process (Windows)
taskkill /PID <PID> /F

# Or change port in backend_server.py
```

#### Detection Script Can't Connect

**Symptom:** "Failed to connect to WebSocket"

**Check:**
1. Backend server is running
2. URL is correct (`ws://localhost:8000/device`)
3. No firewall blocking port 8000

**Solution:**
```bash
# Verify backend is running
python backend_server.py

# Check connection with test script
python -c "import websockets; import asyncio; asyncio.run(websockets.connect('ws://localhost:8000/device'))"
```

#### No Video Stream in Dashboard

**Symptom:** Device card shows but no video

**Check:**
1. Detection running with `--enable-streaming` flag
2. Backend shows "Received video frame from..."
3. Browser console shows frame messages

**Solution:**
```bash
# Ensure streaming is enabled
python run.py --source 0 --enable-streaming

# Check backend logs for frame messages
# Check browser console (F12) for "[WebSocket] 📹 Frame from..."
```

### Frontend Issues

#### Dashboard Shows "Disconnected"

**Symptom:** Red "Disconnected" banner

**Solution:**
1. Verify backend server is running
2. Hard refresh browser (Ctrl+Shift+R)
3. Check browser console for errors
4. Clear browser cache

#### No Devices Showing

**Symptom:** "No Devices Connected" message

**Check:**
1. Detection script is running
2. Detection script connected to backend
3. Backend shows "Detection device connected"

**Solution:**
```bash
# Restart detection with correct flags
python run.py --source 0 --enable-streaming

# Check backend terminal for "Detection device connected"
```

#### Video Lag or Stuttering

**Symptom:** Video updates slowly

**Solution:**
```bash
# Reduce frame rate
python run.py --source 0 --enable-streaming --max-frame-rate 5

# Reduce quality
python run.py --source 0 --enable-streaming --jpeg-quality 70

# Reduce resolution
python run.py --source 0 --enable-streaming --max-frame-width 640
```

### Performance Issues

#### Low FPS

**Solutions:**
1. Use ONNX model: `--onnx-model models/yolov8s.onnx`
2. Use smaller model: `yolov8n.pt` instead of `yolov8s.pt`
3. Increase confidence: `--confidence 0.7`
4. Use ROI to limit detection area
5. Reduce video resolution

#### High CPU Usage

**Solutions:**
1. Reduce ONNX threads: Edit `config.json` → `inter_op_num_threads: 2`
2. Lower frame rate: `--max-frame-rate 5`
3. Use ROI to reduce processing area

#### High Memory Usage

**Solutions:**
1. Reduce buffer sizes in `config.json`
2. Lower streaming quality: `--jpeg-quality 70`
3. Use smaller model: `yolov8n.pt`

### Common Errors

#### "Camera not found"

```bash
# List available cameras (Windows)
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"

# Try different indices
python run.py --source 0 --enable-streaming
python run.py --source 1 --enable-streaming
```

#### "Model file not found"

```bash
# Verify model exists
ls models/yolov8s.pt

# Download if missing (will auto-download on first run)
python run.py --source 0
```

#### "ROI file invalid"

```bash
# Validate JSON
python -c "import json; print(json.load(open('examples/roi_corridor.json')))"

# Check format matches specification
```

## 📊 Performance Benchmarks

### Hardware Performance

| Hardware | Model | FPS | Notes |
|----------|-------|-----|-------|
| Intel i7 (8 cores) | PyTorch | 25-30 | CPU only |
| Intel i7 (8 cores) | ONNX | 30-35 | CPU optimized |
| Orange Pi 5 | PyTorch | 3-5 | ARM CPU |
| Orange Pi 5 | ONNX | 10-15 | ARM optimized |
| NVIDIA RTX 3060 | PyTorch | 60+ | GPU accelerated |

### Streaming Performance

| Resolution | Quality | FPS | Bandwidth | Latency |
|------------|---------|-----|-----------|---------|
| 640x480 | 70 | 10 | ~200 KB/s | <100ms |
| 1280x720 | 85 | 10 | ~400 KB/s | <150ms |
| 1920x1080 | 90 | 10 | ~800 KB/s | <200ms |

## 📚 Additional Documentation

- **[QUICK_START.md](QUICK_START.md)** - 3-step quick start guide
- **[STREAMING_GUIDE.md](STREAMING_GUIDE.md)** - Complete streaming setup
- **[ROI_GUIDE.md](ROI_GUIDE.md)** - ROI configuration guide
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Detailed troubleshooting
- **[SUCCESS_GUIDE.md](SUCCESS_GUIDE.md)** - What success looks like
- **[DEBUG_NO_DEVICES.md](DEBUG_NO_DEVICES.md)** - Debug device connection issues

## 🔐 Security Considerations

### Production Deployment

1. **Use WSS (WebSocket Secure)** instead of WS
2. **Implement authentication** for WebSocket connections
3. **Rate limiting** to prevent abuse
4. **Input validation** on all messages
5. **CORS configuration** for frontend
6. **Firewall rules** for port 8000

### Network Security

```bash
# Use secure WebSocket
WS_URL=wss://your-domain.com/device

# Bind to specific interface (not 0.0.0.0)
# Edit backend_server.py: host = "127.0.0.1"
```

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **YOLOv8** by Ultralytics
- **OpenCV** community
- **React** and **TypeScript** communities
- **WebSocket** protocol contributors

## 📞 Support

For issues or questions:

1. Check documentation files
2. Review troubleshooting guides
3. Check browser console (F12) for errors
4. Check backend logs for errors
5. Create GitHub issue with:
   - System info (OS, Python version, Node version)
   - Command used
   - Error messages
   - Logs from backend and frontend

---

**Made with ❤️ for real-time computer vision monitoring**
