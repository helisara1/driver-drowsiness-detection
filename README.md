# Driver Drowsiness Detection System 🚗😴

A **real-time computer vision system** that detects driver fatigue using **eye closure detection and yawn detection**.
The system monitors the driver's face through a webcam and triggers an **audio alert** when signs of drowsiness are detected.

---

## Features ✨

* CNN-based **eye open/closed detection**
* CNN-based **yawn detection**
* **Hybrid fusion logic** using Eye Aspect Ratio (EAR) + Mouth Aspect Ratio (MAR)
* **Real-time webcam inference**
* **Audio alert system** when drowsiness is detected

---

## Tech Stack 🛠

* **Python**
* **TensorFlow / Keras**
* **OpenCV**
* **dlib**
* **NumPy**

---

## Project Structure 📂

```
driver-drowsiness-detection/
│
├── src/
│   ├── train_eye_model.py
│   ├── train_yawn_model.py
│   └── detect_drowsiness.py
│
├── haarcascades/
│   ├── haarcascade_frontalface_default.xml
│   └── haarcascade_mcs_mouth.xml
│
├── assets/
│   └── alarm.wav
│
├── notebooks/
│   └── proje1.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset 📊

The dataset used for training is the **MRL Eye Dataset**.

Dataset source:
https://www.kaggle.com/datasets/serenaraju/mrl-eye-dataset

⚠️ Dataset is **not included in this repository** due to GitHub size limits.

---

## Installation ⚙️

Clone the repository:

```bash
git clone https://github.com/yourusername/driver-drowsiness-detection.git
cd driver-drowsiness-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training Models 🧠

Train the eye detection model:

```bash
python src/train_eye_model.py
```

Train the yawn detection model:

```bash
python src/train_yawn_model.py
```

---

## Run Real-Time Detection 🎥

```bash
python src/detect_drowsiness.py
```

The webcam will start and the system will monitor the driver's eyes and mouth.
If drowsiness is detected, an **alarm sound** will be triggered.

---

## Results 📈

| Model                | Accuracy |
| -------------------- | -------- |
| Eye Detection Model  | ~94%     |
| Yawn Detection Model | ~71%     |
| Real-time speed      | ~20 FPS  |

---

## Future Improvements 🚀

* Build a **REST API backend**
* Add **web dashboard for monitoring**
* Improve **low-light performance**
* Optimize for **embedded systems (Raspberry Pi / Jetson Nano)**

---

## Author 👨‍💻

Your Name
GitHub: https://github.com/yourusername
