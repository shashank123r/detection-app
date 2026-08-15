# Live Object Detection on Live Streams or Videos — Flask Web Application using YOLOv8

A Flask-based web application that processes live video streams, DroidCam phone cameras, RTSP streams, or webcam feeds and performs real-time object detection using Ultralytics YOLOv8.

## Features

- **Multiple Video Sources** — Webcam index (0, 1...), DroidCam HTTP URL (`http://192.168.x.x:4747/video`), RTSP stream (`rtsp://...`), or any URL supported by OpenCV
- **Real-Time Object Detection** — Using YOLOv8 (80 COCO classes) with configurable confidence threshold
- **Class Filtering** — Select specific object classes to detect from a list of 26 common classes (person, car, cell phone, bottle, etc.), or detect all 80 COCO classes
- **MJPEG Live Stream** — Low-latency video feed displayed in the browser
- **FPS Counter** — Real-time performance monitoring
- **Per-Class Object Counts** — See exactly how many objects of each type are detected
- **Toggle Detection** — Enable/disable detection on the fly to see raw vs. processed feed
- **Snapshots** — Capture and save frames with one click; recent snapshots gallery in the sidebar
- **Health & Debug Endpoints** — `/health`, `/ping` for monitoring, request logging for debugging
- **Auto-Open Browser** — Automatically opens the app on startup
- **Responsive UI** — Works on desktop and mobile with a clean, modern design

## Supported Video Sources

| Type | Example | Description |
|------|---------|-------------|
| Webcam | `0` | Built-in or USB camera index |
| DroidCam | `http://192.168.1.10:4747/video` | Use your phone as a wireless camera via DroidCam app (Android/iOS) — connect via USB or WiFi |
| RTSP | `rtsp://user:pass@192.168.1.100:554/stream` | IP cameras, CCTV, or any RTSP source |
| IP Camera | `http://admin:pass@192.168.1.100/video` | MJPEG/HTTP camera streams |
| Video File | `/path/to/video.mp4` | Local video file path |

## Prerequisites

- Python 3.10+
- pip (Python package manager)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/shashank123r/detection-app.git
   cd detection-app
   ```

2. Install PyTorch (required by Ultralytics YOLOv8):

   **CPU only:**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

   **With CUDA (GPU)** — replace `cu121` with your CUDA version:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

3. Install remaining dependencies:

   ```bash
   pip install flask ultralytics opencv-python numpy
   ```

## Usage

1. Run the Flask application:

   ```bash
   python app.py
   ```

2. The app starts on **`http://127.0.0.1:8080`** and automatically opens in your browser.

3. Enter a video source:
   - Type `0` for your webcam
   - Or a DroidCam URL like `http://192.168.1.10:4747/video`
   - Or an RTSP stream like `rtsp://192.168.1.100:554/stream`

4. Adjust the **Confidence Threshold** slider (0.10 – 0.90).

5. Optionally check specific **object classes** to filter detections (leave empty to detect all 80 COCO classes).

6. Click **"Start Stream"** to begin.

7. On the stream page:
   - Toggle **Detection** on/off
   - Click **Take Snapshot** to capture the current frame
   - View real-time **FPS** and **object counts** in the sidebar
   - Browse **Recent Snapshots** in the gallery

8. Click **"New Stream"** to go back and change the source.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET/POST | Home page with source input form / Start a stream |
| `/stream` | GET | Live stream page with video feed & controls |
| `/video_feed` | GET | MJPEG video stream (multipart/x-mixed-replace) |
| `/stats` | GET | JSON: FPS, detection status, per-class object counts |
| `/toggle_detection` | POST | Toggle object detection on/off |
| `/snapshot` | POST | Save a snapshot of the current frame |
| `/snapshots` | GET | List recent snapshots (up to 24) |
| `/health` | GET | Health check JSON (`{"status": "ok"}`) |
| `/ping` | GET | Plain-text ping (`OK`) — use if HTML pages don't load |

## How It Works

### Backend (`app.py`)

1. **Startup** — Loads `yolov8n.pt` on boot (auto-downloaded on first run) and starts Flask on port 8080.
2. **StreamManager** — A thread-safe class that manages a single video capture pipeline:
   - Opens the video source via `cv2.VideoCapture`
   - Runs a background thread that continuously reads frames
   - If detection is enabled, runs YOLO prediction on each frame and applies class filters
   - Tracks FPS and per-class object counts
   - Maintains a small frame buffer for MJPEG streaming
3. **MJPEG Streaming** — `/video_feed` serves frames as `multipart/x-mixed-replace` for low-latency browser display.
4. **Snapshots** — Frames are saved as JPEG to `static/snapshots/` with timestamps.

### Frontend (`templates/index.html`)

- **Home Page** — Centered card with the source input form, confidence slider, and class filter checkboxes.
- **Stream Page** — Two-column layout: video feed on the left, sidebar on the right with stats, controls, detected objects list, and snapshot gallery.
- **JavaScript** — Polls `/stats` every 500ms for real-time updates. Handles toggle and snapshot via fetch API.
- **Responsive** — Stacks vertically on screens < 900px.

## Project Structure

```
detection-app/
├── app.py                  # Main Flask application with StreamManager
├── camera_settings.py      # Camera settings utility (exposure, contrast)
├── requirements.txt        # Python dependencies
├── run_test.py             # Quick endpoint test script
├── yolov8n.pt              # Pre-trained YOLOv8 model (auto-downloaded)
├── templates/
│   └── index.html          # Single-page frontend (home + stream)
└── static/
    └── snapshots/          # Captured snapshots (created at runtime)
```

## DroidCam Setup (Phone as Webcam)

1. Install the **DroidCam** app on your phone (Android or iOS).
2. Connect your phone and computer to the same WiFi network.
3. Open DroidCam and note the IP address and port shown (e.g., `192.168.1.10:4747`).
4. In the app, enter: `http://192.168.1.10:4747/video` (use your phone's actual IP).
5. Click **"Start Stream"** — your phone camera becomes the video source for object detection.

## Testing

Run the quick test script:

```bash
python run_test.py
```

This starts the server on port 5001 and tests the `/ping`, `/health`, and `/` endpoints.

## Technologies Used

- **Python 3** + **Flask** — Web framework
- **OpenCV (cv2)** — Video capture and frame processing
- **YOLOv8 (Ultralytics)** — Object detection deep learning model
- **PyTorch** — Deep learning backend for YOLOv8
- **NumPy** — Array operations
- **Threading** — Background capture loop
- **HTML/CSS/JavaScript** — Frontend with clean, responsive UI

---

**Created by [Shashank R](https://github.com/shashank123r)**
