import cv2
import threading
import simpleaudio as sa
from ultralytics import YOLO
import time
import requests

ESP_IP = "http://10.231.11.154"   # CHANGE THIS

# =========================
# LOAD MODEL
# =========================
try:
    model = YOLO("best.pt")
    print("Custom model loaded")
except:
    model = YOLO("yolov8n.pt")
    print("Default model loaded")

# =========================
# SOUND
# =========================
def play_alert():
    try:
        sa.WaveObject.from_wave_file("alert.wav").play()
    except:
        pass

# =========================
# DROWSINESS
# =========================
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

eye_closed_frames = 0
EYE_THRESHOLD = 15

cap = cv2.VideoCapture(0)

last_alert_time = 0
cooldown = 3

while True:
    ret, frame = cap.read()
    if not ret:
        break

    danger = False
    detected = set()

    results = model(frame)[0]

    for box in results.boxes:
        label = model.names[int(box.cls[0])].lower()
        detected.add(label)

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
        cv2.putText(frame, label, (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # SAFETY
    if "person" in detected and "helmet" not in detected:
        danger = True

    if "person" in detected and "vest" not in detected:
        danger = True

    if "cell phone" in detected or "mobile" in detected:
        danger = True

    if "fire" in detected:
        danger = True

    # DROWSINESS
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) == 0:
            eye_closed_frames += 1
        else:
            eye_closed_frames = 0

        if eye_closed_frames > EYE_THRESHOLD:
            danger = True

    # =========================
    # SEND TO ESP32
    # =========================
    try:
        if danger:
            requests.get(f"{ESP_IP}/alert")
        else:
            requests.get(f"{ESP_IP}/safe")
    except:
        print("ESP32 not reachable")

    # SOUND
    if danger:
        current_time = time.time()
        if current_time - last_alert_time > cooldown:
            threading.Thread(target=play_alert).start()
            last_alert_time = current_time

    cv2.imshow("AI Safety", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()