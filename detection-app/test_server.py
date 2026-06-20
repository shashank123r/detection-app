"""Quick test to verify the Flask application starts and serves the homepage correctly."""
import os
import sys
import time
import threading
import urllib.request
import urllib.error

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Kill any process on port 5001
os.system("netstat -ano | findstr :5001 > nul 2>&1")

from app import app

def run_server():
    app.run(debug=False, threaded=True, host="127.0.0.1", port=5001, use_reloader=False)

t = threading.Thread(target=run_server, daemon=True)
t.start()

time.sleep(3)  # Wait for server to start

# Test the homepage
try:
    resp = urllib.request.urlopen("http://127.0.0.1:5001/", timeout=10)
    content = resp.read().decode("utf-8")
    status = resp.status
    print(f"STATUS: {status}")
    print(f"CONTENT_LENGTH: {len(content)} bytes")
    print(f"FIRST_500_CHARS: {content[:500]}")
    print(f"HAS_FORM: {'<form' in content}")
    print(f"HAS_YOLO_TITLE: {'YOLOv8' in content}")
    print(f"HAS_SHOW_STREAM: {'show_stream' in content}")
    print("--- HOMEPAGE LOADS OK ---")
except Exception as e:
    print(f"ERROR: {e}")
