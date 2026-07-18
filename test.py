from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch
from collections import deque   # NEW

marked_names = set()
detection_times = {}
THRESHOLD_TIME = 60

# ---------------- VIDEO SETTINGS ----------------
fps = 20
frame_width = 640
frame_height = 480

fourcc = cv2.VideoWriter_fourcc(*'XVID')

first_video = cv2.VideoWriter('first_10min.avi', fourcc, fps, (frame_width, frame_height))
last_video = cv2.VideoWriter('last_10min.avi', fourcc, fps, (frame_width, frame_height))

FIRST_DURATION = 12  # 10 minutes

# Buffer for last 10 minutes
buffer_size = fps * 12
frame_buffer = deque(maxlen=buffer_size)

start_time = time.time()
# ------------------------------------------------


def speak(str1):
    try:
        speak = Dispatch("SAPI.SpVoice")
        speak.Speak(str1)
    except Exception as e:
        print("Speech error:", e)


video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)

with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

print('Shape of Faces matrix --> ', FACES.shape)

knn = KNeighborsClassifier(n_neighbors=5)

min_len = min(len(FACES), len(LABELS))
FACES = FACES[:min_len]
LABELS = LABELS[:min_len]

knn.fit(FACES, LABELS)

imgBackground = cv2.imread("background.png")

COL_NAMES = ['NAME', 'TIME']

if not os.path.exists("Attendance"):
    os.makedirs("Attendance")

today_date = datetime.now().strftime("%d-%m-%Y")
attendance_file = f"Attendance/Attendance_{today_date}.csv"

if not os.path.isfile(attendance_file):
    with open(attendance_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(COL_NAMES)


while True:

    ret, frame = video.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640,480))

    runtime = time.time() - start_time
    if runtime <= FIRST_DURATION:
        first_video.write(frame)

    frame_buffer.append(frame.copy())

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

        output = knn.predict(resized_img)
        name = str(output[0])

        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)

        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_COMPLEX, 1,
                    (255, 255, 255), 2)

        current_time = time.time()

        if name not in detection_times:
            detection_times[name] = current_time

        elapsed_time = current_time - detection_times[name]
        remaining = int(THRESHOLD_TIME - elapsed_time)

        if remaining > 0:
            cv2.putText(frame, f"Stay {remaining}s",
                        (x, y+h+30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0,255,0), 2)

        if elapsed_time >= THRESHOLD_TIME and name not in marked_names:

            ts = time.time()
            timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

            with open(attendance_file, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([name, timestamp])

            marked_names.add(name)

            speak(f"Attendance marked for {name}")
            print(f"Attendance marked for {name} at {timestamp}")

    try:
        imgBackground[162:162+480, 55:55+640] = frame
        cv2.imshow("Attendance System", imgBackground)

    except:
        cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) == ord('q'):
        break


# -------- SAVE LAST 10 MINUTES VIDEO --------
for f in frame_buffer:
    last_video.write(f)

video.release()
first_video.release()
last_video.release()

cv2.destroyAllWindows()