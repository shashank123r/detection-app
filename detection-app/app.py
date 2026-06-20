from ultralytics import YOLO
import cv2
import numpy as np
from flask import Flask, render_template, request, Response, jsonify, redirect, url_for
import threading
import time
import os
from datetime import datetime
import collections

app = Flask(__name__)

# ── Request logging (for debugging) ──
@app.before_request
def log_request():
    """Log every incoming request."""
    print(f"[REQ] {request.method} {request.path}", flush=True)

@app.after_request
def log_response(response):
    """Log every outgoing response (skip streaming responses)."""
    if not response.is_streamed:
        print(f"[RES] {request.method} {request.path} -> {response.status_code} ({response.content_length or '?'} bytes)", flush=True)
    else:
        print(f"[RES] {request.method} {request.path} -> {response.status_code} (streaming)", flush=True)
    return response


# Load YOLO model (auto-downloads yolov8n.pt from ultralytics)
print("[BOOT] Loading YOLO model...", flush=True)
model = YOLO("yolov8n.pt")
print("[BOOT] YOLO model loaded successfully", flush=True)

# Common COCO classes for the filter checkboxes
COMMON_FILTER_CLASSES = [
    "person", "car", "cell phone", "bottle", "laptop",
    "book", "chair", "dog", "cat", "tv", "bicycle",
    "motorcycle", "bus", "truck", "bird", "cup",
    "wine glass", "pizza", "donut", "cake", "mouse",
    "keyboard", "remote", "backpack", "suitcase", "umbrella"
]


class StreamManager:
    """Manages a single camera capture thread and detection pipeline."""

    def __init__(self):
        self.lock = threading.Lock()
        self.cap = None
        self.running = False
        self.detection_enabled = True
        self.confidence = 0.5
        self.selected_classes = None  # None = detect all
        self.buffer = collections.deque(maxlen=2)
        self.raw_frame = None
        self.last_results = None
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()
        self.bg_thread = None
        self.snapshot_dir = os.path.join("static", "snapshots")

    def start(self, source, confidence=0.5, selected_classes=None):
        """Start capturing from the given source."""
        self.stop()
        try:
            src = int(source) if source.isdigit() else source
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                return False
        except (ValueError, cv2.error):
            return False

        with self.lock:
            self.cap = cap
            self.source = source
            self.confidence = confidence
            self.selected_classes = selected_classes if selected_classes else None
            self.running = True
            self.detection_enabled = True
            self.frame_count = 0
            self.last_time = time.time()
            self.fps = 0
            self.raw_frame = None
            self.last_results = None
            self.buffer.clear()

        self.bg_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.bg_thread.start()
        return True

    def stop(self):
        """Stop capture and release resources."""
        with self.lock:
            self.running = False
            if self.cap:
                self.cap.release()
                self.cap = None

    def _capture_loop(self):
        """Background thread: continuously reads frames and runs detection."""
        while True:
            with self.lock:
                if not self.running or self.cap is None:
                    break
                ret, frame = self.cap.read()
                if not ret:
                    self.running = False
                    break

                # Update FPS counter
                self.frame_count += 1
                elapsed = time.time() - self.last_time
                if elapsed >= 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.last_time = time.time()

                self.raw_frame = frame.copy()

                if self.detection_enabled:
                    # Build reverse mapping from name to class index
                    name_to_idx = {v: k for k, v in model.names.items()}
                    # Filter classes if selected
                    if self.selected_classes:
                        class_ids = [name_to_idx[c] for c in self.selected_classes
                                     if c in name_to_idx]
                        results = model.predict(
                            frame, conf=self.confidence, verbose=False,
                            classes=class_ids if class_ids else None
                        )
                    else:
                        results = model.predict(frame, conf=self.confidence, verbose=False)
                    self.last_results = results
                    display = results[0].plot()
                else:
                    self.last_results = None
                    display = frame.copy()

                _, jpg = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 85])
                self.buffer.append(jpg.tobytes())

            time.sleep(0.001)

    def generate_frames(self):
        """Generator that yields JPEG frames for MJPEG streaming."""
        while self.running:
            with self.lock:
                frame_bytes = self.buffer[-1] if self.buffer else None
            if frame_bytes:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.03)

    def toggle_detection(self):
        """Toggle detection on/off. Returns new state."""
        with self.lock:
            self.detection_enabled = not self.detection_enabled
            return self.detection_enabled

    def snapshot(self):
        """Save the current raw frame as a snapshot. Returns filename or None."""
        with self.lock:
            if self.raw_frame is None:
                return None
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"snapshot_{ts}.jpg"
            cv2.imwrite(os.path.join(self.snapshot_dir, filename), self.raw_frame)
            return filename

    def get_stats(self):
        """Return current FPS and per-class object counts."""
        with self.lock:
            counts = {}
            if self.last_results is not None and self.detection_enabled:
                boxes = self.last_results[0].boxes
                if boxes is not None and boxes.cls is not None:
                    names = model.names
                    for cls_id in boxes.cls.tolist():
                        name = names[int(cls_id)]
                        counts[name] = counts.get(name, 0) + 1
            return {
                "fps": round(self.fps, 1),
                "detection_enabled": self.detection_enabled,
                "class_counts": counts
            }

    def update_settings(self, confidence=None):
        """Update runtime settings."""
        with self.lock:
            if confidence is not None:
                self.confidence = confidence


manager = StreamManager()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():
    """Home page with URL input form, or stream page after submission."""
    if request.method == "POST":
        source = request.form.get("source", "").strip()
        confidence = float(request.form.get("confidence", 0.5))
        selected = request.form.getlist("classes")
        selected = selected if selected else None

        if not source:
            return render_template("index.html", error="Please enter a video source or camera index",
                                   filter_classes=COMMON_FILTER_CLASSES)

        if not manager.start(source, confidence, selected):
            return render_template("index.html",
                                   error=f"Could not open video source: {source}. "
                                         f"Try a DroidCam URL like http://192.168.x.x:4747/video, "
                                         f"rtsp://..., or a camera index (0, 1, ...)",
                                   filter_classes=COMMON_FILTER_CLASSES)
        return redirect(url_for("stream"))

    # Show home page
    return render_template("index.html", filter_classes=COMMON_FILTER_CLASSES)


@app.route("/stream")
def stream():
    """Show the live stream page."""
    if not manager.running:
        return redirect(url_for("home"))
    source_label = str(manager.source) if not isinstance(manager.source, str) else manager.source
    return render_template("index.html", show_stream=True,
                           filter_classes=COMMON_FILTER_CLASSES,
                           source=source_label)


@app.route("/video_feed")
def video_feed():
    """MJPEG video stream."""
    if not manager.running:
        return "", 204
    return Response(manager.generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats")
def stats():
    """JSON endpoint for FPS and per-class counts."""
    return jsonify(manager.get_stats())


@app.route("/toggle_detection", methods=["POST"])
def toggle_detection():
    """Toggle object detection on/off."""
    return jsonify({"detection_enabled": manager.toggle_detection()})


@app.route("/snapshot", methods=["POST"])
def take_snapshot():
    """Save a snapshot and return its URL."""
    fn = manager.snapshot()
    if fn is None:
        return jsonify({"error": "No frame available yet"}), 400
    return jsonify({
        "filename": fn,
        "url": url_for("static", filename=f"snapshots/{fn}")
    })


@app.route("/snapshots")
def list_snapshots():
    """Return list of recent snapshots."""
    try:
        files = sorted(os.listdir(manager.snapshot_dir), reverse=True)[:24]
        return jsonify({
            "snapshots": [
                {"name": f, "url": url_for("static", filename=f"snapshots/{f}")}
                for f in files
            ]
        })
    except FileNotFoundError:
        return jsonify({"snapshots": []})


@app.route("/health")
def health():
    """Simple health check endpoint."""
    return jsonify({"status": "ok", "running": manager.running})


@app.route("/ping")
def ping():
    """Plain-text ping endpoint - use this if HTML pages don't load."""
    return "OK\n", 200, {"Content-Type": "text/plain"}


@app.route("/favicon.ico")
def favicon():
    """Prevent 404 for favicon."""
    return "", 204


@app.errorhandler(404)
def not_found(e):
    return "404 - Page not found. Try /, /health, /ping\n", 404, {"Content-Type": "text/plain"}


@app.errorhandler(500)
def server_error(e):
    return "500 - Internal server error. Check the terminal for error details.\n", 500, {"Content-Type": "text/plain"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("static/snapshots", exist_ok=True)
    
    PORT = 8080
    URL = f"http://127.0.0.1:{PORT}"
    
    print("=" * 50, flush=True)
    print(f"  YOLOv8 Object Detection Server", flush=True)
    print(f"  URL: {URL}", flush=True)
    print(f"  Test: {URL}/ping", flush=True)
    print("=" * 50, flush=True)
    
    # Open browser automatically after a short delay
    def open_browser():
        time.sleep(1.5)
        try:
            import webbrowser
            webbrowser.open(URL)
            print(f"[BOOT] Browser opened to {URL}", flush=True)
        except Exception as e:
            print(f"[BOOT] Could not open browser automatically: {e}", flush=True)
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(debug=True, threaded=True, host="0.0.0.0", port=PORT, use_reloader=False)
