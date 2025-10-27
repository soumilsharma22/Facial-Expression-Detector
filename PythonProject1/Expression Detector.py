"""
smart_emotion_detector_optimized.py - FIXED VERSION
"""

import cv2
import numpy as np
import os
from datetime import datetime
import pandas as pd
from deepface import DeepFace
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# ----------------- CONFIG -----------------
DB_PATH = "faces_db/"            # folder for recognized faces (optional)
LOG_FILE = "emotion_log.csv"
ALERT_EMOTIONS = {'sad', 'angry'}
ALERT_SOUND = "alert.mp3"        # place a short mp3 or leave blank to disable
SKIP_FRAMES = 5                  # process one frame every SKIP_FRAMES (>=1). Increase for speed
MAX_WORKERS = 1                  # start with 1 worker for stability
DETECTOR_SCALE = 0.5             # scale frame to this fraction for detection speed (0.5 = half size)
FRAME_WIDTH = 640                # resize camera frame to this width (keeps aspect ratio)
USE_FACE_RECOGNITION = False     # set True to enable DeepFace.find (slower)
DETECTION_BACKEND = "opencv"     # DeepFace detector backend when used on ROIs
MIN_FACE_SIZE = 80               # minimum face size in pixels to analyze
# ------------------------------------------

# Create log if not exists
if not os.path.exists(LOG_FILE):
    df = pd.DataFrame(columns=["Timestamp", "Person", "Emotion", "Age", "Gender"])
    df.to_csv(LOG_FILE, index=False)
    print(f"Created log file: {LOG_FILE}")

# Load Haar cascade for fast face detection
haar_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_xml)
if face_cascade.empty():
    raise RuntimeError("Failed to load Haar cascade. Check your OpenCV installation.")

print("Initializing DeepFace...")
# Test DeepFace with a simple call first
try:
    # Create a test image to verify DeepFace works
    test_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    test_img[30:70, 30:70] = 0

    # This should fail (no face) but test if DeepFace loads properly
    DeepFace.analyze(test_img, actions=['emotion'], enforce_detection=False, silent=True)
    print("DeepFace initialized successfully")
except Exception as e:
    print(f"DeepFace test failed: {e}")
    print("Trying to continue anyway...")

def identify_face_simple(face_img):
    """
    Simplified face analysis with robust error handling
    """
    out = {
        "emotion": "unknown",
        "age": "unknown",
        "gender": "unknown",
        "name": "Unknown",
        "error": None,
        "debug": "started"
    }

    # Validate input image
    if face_img is None or face_img.size == 0:
        out["error"] = "Empty image"
        return out

    if face_img.shape[0] < 50 or face_img.shape[1] < 50:
        out["error"] = f"Image too small: {face_img.shape}"
        return out

    try:
        out["debug"] = "converting to RGB"
        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # Save debug image to see what we're analyzing
        debug_path = "debug_face.jpg"
        cv2.imwrite(debug_path, face_img)
        out["debug"] = f"saved debug image to {debug_path}"

        # Try simple analysis with minimal parameters
        out["debug"] = "calling DeepFace.analyze"
        analysis = DeepFace.analyze(
            rgb_face,
            actions=['emotion', 'age', 'gender'],
            enforce_detection=False,  # CRITICAL: don't fail if no face detected
            detector_backend="opencv",
            silent=True,
            align=False  # Try without alignment
        )

        out["debug"] = "DeepFace analysis completed"

        # Handle response format
        if isinstance(analysis, list):
            analysis = analysis[0]

        # Extract results with fallbacks
        out["emotion"] = analysis.get('dominant_emotion', 'unknown')
        out["age"] = analysis.get('age', 'unknown')

        # Handle gender field (DeepFace is inconsistent)
        gender = analysis.get('dominant_gender', analysis.get('gender', 'unknown'))
        out["gender"] = gender

        out["debug"] = "success"

    except Exception as e:
        out["error"] = f"Analysis failed: {str(e)}"
        out["debug"] = f"exception: {type(e).__name__}"

    return out

def identify_face_even_simpler(face_img):
    """
    Even simpler version - try just emotion detection first
    """
    out = {
        "emotion": "unknown",
        "age": "unknown",
        "gender": "unknown",
        "name": "Unknown",
        "error": None
    }

    if face_img is None or face_img.size == 0:
        out["error"] = "Empty image"
        return out

    try:
        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # Try JUST emotion first
        emotion_analysis = DeepFace.analyze(
            rgb_face,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend="opencv",
            silent=True,
            align=False
        )

        if isinstance(emotion_analysis, list):
            emotion_analysis = emotion_analysis[0]

        out["emotion"] = emotion_analysis.get('dominant_emotion', 'unknown')

        # If emotion worked, try age and gender
        try:
            full_analysis = DeepFace.analyze(
                rgb_face,
                actions=['age', 'gender'],
                enforce_detection=False,
                detector_backend="opencv",
                silent=True,
                align=False
            )

            if isinstance(full_analysis, list):
                full_analysis = full_analysis[0]

            out["age"] = full_analysis.get('age', 'unknown')
            out["gender"] = full_analysis.get('dominant_gender', full_analysis.get('gender', 'unknown'))

        except Exception as e:
            # It's OK if age/gender fails, we at least have emotion
            print(f"Age/gender analysis failed (but emotion worked): {e}")

    except Exception as e:
        out["error"] = f"All analysis failed: {str(e)}"

    return out

# Use the simpler version
identify_face = identify_face_even_simpler

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    # Try different camera indices
    for i in range(1, 4):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Found camera at index {i}")
            break
    else:
        raise RuntimeError("Cannot open any camera.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)

frame_count = 0
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
futures = []
last_debug_info = ""

print("Starting camera. Press 'q' to quit, 'd' to show debug info.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Resize frame
        height, width = frame.shape[:2]
        new_w = FRAME_WIDTH
        scale = new_w / float(width)
        new_h = int(height * scale)
        frame = cv2.resize(frame, (new_w, new_h))
        display_frame = frame.copy()

        # Only run detection every SKIP_FRAMES frames
        if frame_count % SKIP_FRAMES == 0:
            # Small grayscale frame for detection
            small = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(small, (0, 0), fx=DETECTOR_SCALE, fy=DETECTOR_SCALE)

            faces = face_cascade.detectMultiScale(
                small,
                scaleFactor=1.1,
                minNeighbors=6,  # Increased for fewer false positives
                minSize=(40, 40)
            )

            # Scale face coordinates back to original frame size
            faces_scaled = []
            for (x, y, w, h) in faces:
                x = int(x / DETECTOR_SCALE)
                y = int(y / DETECTOR_SCALE)
                w = int(w / DETECTOR_SCALE)
                h = int(h / DETECTOR_SCALE)

                if w >= MIN_FACE_SIZE and h >= MIN_FACE_SIZE:
                    faces_scaled.append((x, y, w, h))

            # Submit analysis jobs for detected faces
            if len(futures) < MAX_WORKERS:  # Be more conservative
                for (x, y, w, h) in faces_scaled:
                    # Create ROI with padding but be careful about boundaries
                    pad = int(0.1 * min(w, h))  # Reduced padding
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(frame.shape[1], x + w + pad)
                    y2 = min(frame.shape[0], y + h + pad)

                    face_roi = frame[y1:y2, x1:x2]

                    # Validate ROI
                    if (face_roi.size == 0 or face_roi.shape[0] < 50 or
                        face_roi.shape[1] < 50 or face_roi.shape[0] > 500):
                        continue

                    # Ensure reasonable aspect ratio (avoid stretched images)
                    h, w = face_roi.shape[:2]
                    aspect_ratio = w / h
                    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                        continue

                    # Submit for analysis
                    future = executor.submit(identify_face, face_roi)
                    future.coords = (x1, y1, x2, y2)
                    future.timestamp = time.time()
                    futures.append(future)

        # Process completed analyses
        current_time = time.time()
        for fut in list(futures):
            # Remove stale futures (older than 5 seconds)
            if current_time - getattr(fut, 'timestamp', 0) > 5.0:
                futures.remove(fut)
                continue

            if fut.done():
                try:
                    res = fut.result(timeout=1.0)
                except Exception as e:
                    res = {"error": f"Future error: {str(e)}"}

                x1, y1, x2, y2 = fut.coords

                # Draw bounding box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                if res.get("error"):
                    label = f"Error: {res.get('error', 'Unknown')}"
                    if 'debug' in res:
                        last_debug_info = res['debug']
                    color = (0, 0, 255)  # Red
                    cv2.putText(display_frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                else:
                    emotion = res.get("emotion", "unknown")
                    age = res.get("age", "unknown")
                    gender = res.get("gender", "unknown")

                    # Use color coding for emotions
                    if emotion in ALERT_EMOTIONS:
                        color = (0, 0, 255)  # Red for alert emotions
                    else:
                        color = (0, 255, 0)   # Green for normal emotions

                    # Main emotion label
                    emotion_text = f"Emotion: {emotion}"
                    cv2.putText(display_frame, emotion_text, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Additional info
                    info_text = f"{gender}, {age}"
                    cv2.putText(display_frame, info_text, (x1, y1 - 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    # Log results
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_row = {
                        "Timestamp": timestamp,
                        "Person": "Unknown",
                        "Emotion": emotion,
                        "Age": age,
                        "Gender": gender
                    }
                    try:
                        pd.DataFrame([log_row]).to_csv(LOG_FILE, mode='a', header=False, index=False)
                    except Exception as e:
                        print("Log write failed:", e)

                futures.remove(fut)

        # Display status information
        status_y = 20
        cv2.putText(display_frame, f"Pending: {len(futures)}", (10, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
        cv2.putText(display_frame, f"Frame: {frame_count}", (10, status_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

        # Show last debug info if available
        if last_debug_info:
            cv2.putText(display_frame, f"Debug: {last_debug_info}", (10, status_y + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow("Smart Emotion Detector - FIXED", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            print(f"Debug: Pending futures: {len(futures)}, Last debug: {last_debug_info}")

        frame_count += 1

finally:
    cap.release()
    cv2.destroyAllWindows()
    executor.shutdown(wait=False)
    print("Exited cleanly.")