"""Start Flask on port 8080 and test the endpoints."""
import os
import sys
import time
import threading
import urllib.request
import urllib.error

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Print the current app.py port setting
with open("app.py", "r") as f:
    content = f.read()
    if "PORT = 8080" in content:
        print("[OK] app.py configured for port 8080")
    else:
        print("[WARN] app.py might not be on port 8080")

from app import app

def run_server():
    app.run(debug=False, threaded=True, host="127.0.0.1", port=8080, use_reloader=False)

t = threading.Thread(target=run_server, daemon=True)
t.start()

time.sleep(3)

tests_passed = 0
tests_failed = 0

# Test /ping
try:
    resp = urllib.request.urlopen("http://127.0.0.1:8080/ping", timeout=10)
    body = resp.read().decode("utf-8")
    assert resp.status == 200
    assert body.strip() == "OK"
    assert "text/plain" in resp.headers.get("Content-Type", "")
    print(f"[PASS] /ping -> 200, body='{body.strip()}'")
    tests_passed += 1
except Exception as e:
    print(f"[FAIL] /ping -> {e}")
    tests_failed += 1

# Test /health
try:
    resp = urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=10)
    body = resp.read().decode("utf-8")
    assert resp.status == 200
    assert "ok" in body
    print(f"[PASS] /health -> 200, body={body}")
    tests_passed += 1
except Exception as e:
    print(f"[FAIL] /health -> {e}")
    tests_failed += 1

# Test / (homepage)
try:
    resp = urllib.request.urlopen("http://127.0.0.1:8080/", timeout=10)
    body = resp.read().decode("utf-8")
    assert resp.status == 200
    assert "YOLOv8" in body
    assert "<form" in body
    print(f"[PASS] / -> 200, {len(body)} bytes, contains YOLOv8 form")
    tests_passed += 1
except Exception as e:
    print(f"[FAIL] / -> {e}")
    tests_failed += 1

print(f"\n=== Results: {tests_passed} passed, {tests_failed} failed ===")
