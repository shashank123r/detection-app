"""Start Flask on port 5001 and test the endpoints."""
import os
import sys
import time
import threading
import urllib.request
import urllib.error

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Import the app
from app import app

def run_server():
    app.run(debug=False, threaded=True, host="127.0.0.1", port=5001, use_reloader=False)

t = threading.Thread(target=run_server, daemon=True)
t.start()

time.sleep(3)

# Test /ping
try:
    resp = urllib.request.urlopen("http://127.0.0.1:5001/ping", timeout=10)
    body = resp.read().decode("utf-8")
    print(f"PING: status={resp.status}, body='{body.strip()}', content-type={resp.headers.get('Content-Type')}")
except Exception as e:
    print(f"PING ERROR: {e}")

# Test /health
try:
    resp = urllib.request.urlopen("http://127.0.0.1:5001/health", timeout=10)
    body = resp.read().decode("utf-8")
    print(f"HEALTH: status={resp.status}, body={body}")
except Exception as e:
    print(f"HEALTH ERROR: {e}")

# Test / (homepage)
try:
    resp = urllib.request.urlopen("http://127.0.0.1:5001/", timeout=10)
    body = resp.read().decode("utf-8")
    print(f"HOME: status={resp.status}, length={len(body)} bytes")
    print(f"HOME contains 'YOLOv8': {'YOLOv8' in body}")
    print(f"HOME contains '<form': {'<form' in body}")
    print(f"HOME first 200 chars: {body[:200]}")
except Exception as e:
    print(f"HOME ERROR: {e}")

print("\n=== ALL TESTS DONE ===")
