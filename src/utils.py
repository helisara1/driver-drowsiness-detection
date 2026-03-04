import cv2
import numpy as np
from scipy.spatial import distance as dist

# ---------- EYE METRIC ----------
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C + 1e-6)

# ---------- MOUTH METRIC ----------
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C + 1e-6)

# ---------- CNN PREPROCESSING ----------
def preprocess_for_cnn(bgr, size=(224,224)):
    try:
        img = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)
    except Exception as e:
        print("Preprocess error:", e)
        return None
