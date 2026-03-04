import os
import cv2
import dlib
import threading
import time
import numpy as np
import pygame
from keras.models import load_model
from scipy.spatial import distance as dist

# Import your helper functions
from utils import eye_aspect_ratio, mouth_aspect_ratio, preprocess_for_cnn

# ---------- PORTABLE PATHS (NO HARD-CODED PATHS) ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EYE_MODEL_PATH  = os.path.join(BASE_DIR, "models", "eye_model.h5")
YAWN_MODEL_PATH = os.path.join(BASE_DIR, "models", "yawn_model.h5")
PREDICTOR_PATH  = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")
ALARM_PATH      = os.path.join(BASE_DIR, "alarm.wav")

# ---------- LOAD MODELS ----------
print("Loading models...")
eye_model  = load_model(EYE_MODEL_PATH)
yawn_model = load_model(YAWN_MODEL_PATH)

detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# ---------- AUDIO ALARM ----------
pygame.mixer.init()
_alarm_lock = threading.Lock()
_alarm_playing = False

def play_alarm_2s():
    global _alarm_playing
    with _alarm_lock:
        if _alarm_playing:
            return
        _alarm_playing = True
    try:
        pygame.mixer.music.load(ALARM_PATH)
        pygame.mixer.music.play()
        time.sleep(2.0)
        pygame.mixer.music.stop()
    except Exception as e:
        print("Alarm error:", e)
    finally:
        with _alarm_lock:
            _alarm_playing = False

# ---------- PARAMETERS ----------
EAR_THRESH = 0.25
MAR_THRESH = 0.60
FRAME_CONSEC = 12
COOLDOWN_S = 5.0

consec_eye = 0
consec_yawn = 0
last_alert_time = 0.0

# ---------- VIDEO LOOP ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("Starting drowsiness detection... Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)
    if len(rects) == 0:
        cv2.imshow("Hybrid Drowsiness Detection (Fusion)", frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    rect = max(rects, key=lambda r: r.width() * r.height())
    shape = predictor(gray, rect)
    pts = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.int32)

    leftEye  = pts[42:48]
    rightEye = pts[36:42]
    mouthPts = pts[48:68]

    ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
    mar = mouth_aspect_ratio(mouthPts)

    cv2.putText(frame, f"EAR:{ear:.2f}", (w-180,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
    cv2.putText(frame, f"MAR:{mar:.2f}", (w-180,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,255), 2)

    # Face ROI
    x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
    x1, y1 = max(0,x1), max(0,y1)
    x2, y2 = min(w,x2), min(h,y2)
    face = frame[y1:y2, x1:x2].copy()

    fh, fw = y2-y1, x2-x1
    eye_roi   = face[int(fh*0.22):int(fh*0.52), int(fw*0.20):int(fw*0.80)]
    mouth_roi = face[int(fh*0.62):int(fh*0.95), int(fw*0.22):int(fw*0.78)]

    # CNN Eye Prediction
    eye_label = 0
    xeye = preprocess_for_cnn(eye_roi)
    if xeye is not None:
        p = eye_model.predict(xeye, verbose=0)[0]
        eye_label = int(np.argmax(p))

    # CNN Mouth Prediction
    yawn_label = 0
    xmouth = preprocess_for_cnn(mouth_roi)
    if xmouth is not None:
        q = yawn_model.predict(xmouth, verbose=0)[0]
        yawn_label = int(np.argmax(q))

    # ---------- Fusion logic ----------
    if (yawn_label == 1) or (mar > MAR_THRESH):
        yawn_label = 1
    else:
        yawn_label = 0

    if (eye_label == 1) or (ear < EAR_THRESH):
        eye_label = 1  # Closed
    else:
        eye_label = 0  # Open
    # ---------------------------------

    cv2.putText(frame, f"Eye: {'Closed' if eye_label==1 else 'Open'}",
                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    cv2.putText(frame, f"Mouth: {'Yawn' if yawn_label==1 else 'No Yawn'}",
                (x1, y2+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    consec_eye  = consec_eye+1 if eye_label==1 else 0
    consec_yawn = consec_yawn+1 if yawn_label==1 else 0

    if (consec_eye >= FRAME_CONSEC) or (consec_yawn >= FRAME_CONSEC):
        if time.time() - last_alert_time > COOLDOWN_S:
            threading.Thread(target=play_alarm_2s).start()
            last_alert_time = time.time()

        cv2.putText(frame, "!!! DROWSINESS ALERT !!!", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)

    cv2.imshow("Hybrid Drowsiness Detection (Fusion)", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
