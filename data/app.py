
import os
import csv
import time
import pickle
import threading
from datetime import datetime

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.neighbors import KNeighborsClassifier
from streamlit_webrtc import webrtc_streamer, WebRtcMode


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Automatic Attendance System",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Automatic Attendance System")
st.caption("Face Recognition Based Attendance System")


# =========================================================
# CONFIGURATION
# =========================================================

THRESHOLD_TIME = 10
MAX_FACE_SAMPLES = 100
FACE_SIZE = (50, 50)

DATA_DIR = "data"
ATTENDANCE_DIR = "attendance"

NAMES_FILE = os.path.join(DATA_DIR, "names.pkl")
FACES_FILE = os.path.join(DATA_DIR, "faces_data.pkl")
CASCADE_FILE = "haarcascade_frontalface_default.xml"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# Separate state for each app session.
if "marked_names" not in st.session_state:
    st.session_state.marked_names = set()

if "model_version" not in st.session_state:
    st.session_state.model_version = 0


# =========================================================
# THREAD-SAFE STORAGE
# =========================================================

file_lock = threading.Lock()
attendance_lock = threading.Lock()

# Shared between video-processing callbacks.
detection_times = {}
marked_names = set()


# =========================================================
# FACE DETECTOR
# =========================================================

facedetect = cv2.CascadeClassifier(CASCADE_FILE)

if facedetect.empty():
    st.error(
        "Face detector file not found. Place "
        "haarcascade_frontalface_default.xml "
        "beside app.py."
    )
    st.stop()


# =========================================================
# WEBRTC CONFIGURATION
# =========================================================

RTC_CONFIGURATION = {
    "iceServers": [
        {
            "urls": ["stun:stun.l.google.com:19302"]
        }
    ]
}


# =========================================================
# LOAD FACE MODEL
# =========================================================

def get_file_version():
    return tuple(
        os.path.getmtime(path) if os.path.exists(path) else 0
        for path in (NAMES_FILE, FACES_FILE)
    )


@st.cache_resource
def load_model(names_version, faces_version, reload_version=0):
    if not os.path.exists(NAMES_FILE):
        return None

    if not os.path.exists(FACES_FILE):
        return None

    try:
        with file_lock:
            with open(NAMES_FILE, "rb") as f:
                labels = pickle.load(f)

            with open(FACES_FILE, "rb") as f:
                faces = pickle.load(f)

        faces = np.asarray(faces)
        labels = np.asarray(labels)

        if faces.ndim != 2 or len(faces) == 0:
            return None

        size = min(len(faces), len(labels))
        faces = faces[:size]
        labels = labels[:size]

        if size == 0:
            return None

        neighbors = min(5, size)

        model = KNeighborsClassifier(
            n_neighbors=neighbors
        )
        model.fit(faces, labels)

        return model

    except Exception as e:
        print("Model loading error:", e)
        return None


def get_model():
    versions = get_file_version()
    return load_model(
        versions[0],
        versions[1],
        st.session_state.model_version
    )


# =========================================================
# ATTENDANCE CSV
# =========================================================

def attendance_file():
    today = datetime.now().strftime("%d-%m-%Y")
    return os.path.join(
        ATTENDANCE_DIR,
        f"Attendance_{today}.csv"
    )


def mark_attendance(name):
    path = attendance_file()
    timestamp = datetime.now().strftime("%H:%M:%S")

    with attendance_lock:
        new_file = not os.path.exists(path)

        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            if new_file:
                writer.writerow(["NAME", "TIME"])

            writer.writerow([name, timestamp])


# =========================================================
# SAVE NEW FACE SAMPLES
# =========================================================

def save_face_data(student_name, samples):
    # Each sample is a 50x50x3 image.
    new_faces = np.asarray(samples).reshape(
        len(samples), -1
    )

    with file_lock:
        if os.path.exists(FACES_FILE):
            with open(FACES_FILE, "rb") as f:
                old_faces = np.asarray(pickle.load(f))

            if old_faces.ndim != 2:
                raise ValueError("Existing face data has an invalid shape.")

            if old_faces.shape[1] != new_faces.shape[1]:
                raise ValueError(
                    "Existing face samples use a different image format."
                )

            new_faces = np.concatenate(
                [old_faces, new_faces], axis=0
            )

        if os.path.exists(NAMES_FILE):
            with open(NAMES_FILE, "rb") as f:
                names = list(pickle.load(f))
        else:
            names = []

        names.extend([student_name] * len(samples))

        # Write temporary files before replacing the originals.
        faces_tmp = FACES_FILE + ".tmp"
        names_tmp = NAMES_FILE + ".tmp"

        with open(faces_tmp, "wb") as f:
            pickle.dump(new_faces, f)

        with open(names_tmp, "wb") as f:
            pickle.dump(names, f)

        os.replace(faces_tmp, FACES_FILE)
        os.replace(names_tmp, NAMES_FILE)


# =========================================================
# VIDEO CALLBACK: ATTENDANCE
# =========================================================

class FaceRecognition:

    def __init__(self, model):
        self.model = model
        self.local_detection_times = {}
        self.local_marked_names = set()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = facedetect.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5
        )

        now = time.monotonic()

        # Forget a detection when a face disappears.
        visible_names = set()

        for (x, y, w, h) in faces:
            crop = img[y:y + h, x:x + w]

            try:
                resized = cv2.resize(crop, FACE_SIZE)
                feature = resized.flatten().reshape(1, -1)

                if self.model is None:
                    cv2.putText(
                        img, "Register faces first",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2
                    )
                    continue

                name = str(self.model.predict(feature)[0])
                visible_names.add(name)

                if name not in self.local_detection_times:
                    self.local_detection_times[name] = now

                elapsed = now - self.local_detection_times[name]
                remaining = max(
                    0, int(THRESHOLD_TIME - elapsed + 0.99)
                )

                cv2.rectangle(
                    img, (x, y), (x + w, y + h),
                    (0, 255, 0), 2
                )

                cv2.putText(
                    img, name, (x, max(25, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2
                )

                if elapsed < THRESHOLD_TIME:
                    cv2.putText(
                        img, f"Hold still: {remaining}s",
                        (x, y + h + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 255, 0), 2
                    )
                elif name not in self.local_marked_names:
                    mark_attendance(name)
                    self.local_marked_names.add(name)

                    cv2.putText(
                        img, "Attendance marked",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2
                    )

            except Exception as e:
                print("Recognition error:", e)

        for name in list(self.local_detection_times):
            if name not in visible_names:
                self.local_detection_times.pop(name, None)

        return av.VideoFrame.from_ndarray(
            img, format="bgr24"
        )


# =========================================================
# VIDEO CALLBACK: ADD NEW FACE
# =========================================================

class AddFace:

    def __init__(self, student_name):
        self.student_name = student_name
        self.samples = []
        self.frame_count = 0
        self.saved = False
        self.save_error = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = facedetect.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5
        )

        for (x, y, w, h) in faces:
            crop = img[y:y + h, x:x + w]

            try:
                resized = cv2.resize(crop, FACE_SIZE)
                self.frame_count += 1

                if (
                    self.frame_count % 5 == 0
                    and len(self.samples) < MAX_FACE_SAMPLES
                    and not self.saved
                ):
                    self.samples.append(resized.copy())

                cv2.rectangle(
                    img, (x, y), (x + w, y + h),
                    (0, 255, 0), 2
                )

                cv2.putText(
                    img, self.student_name,
                    (x, max(25, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2
                )

            except Exception as e:
                print("Face capture error:", e)

        count = len(self.samples)

        cv2.putText(
            img, f"Samples: {count}/{MAX_FACE_SAMPLES}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 255), 2
        )

        if count >= MAX_FACE_SAMPLES and not self.saved:
            try:
                save_face_data(self.student_name, self.samples)
                self.saved = True
                print(f"Successfully registered: {self.student_name}")
            except Exception as e:
                self.save_error = str(e)
                print("Saving face data failed:", e)

        if self.saved:
            cv2.putText(
                img, "FACE SAVED SUCCESSFULLY",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )

        elif self.save_error:
            cv2.putText(
                img, "Save failed - check app logs",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 255), 2
            )

        return av.VideoFrame.from_ndarray(
            img, format="bgr24"
        )


# =========================================================
# MODEL AND TABS
# =========================================================

knn = get_model()

attendance_tab, add_face_tab = st.tabs(
    ["🎥 Attendance", "➕ Add New Face"]
)


# =========================================================
# ATTENDANCE TAB
# =========================================================

with attendance_tab:
    st.subheader("Face Recognition")

    if knn is None:
        st.warning(
            "No trained face model found. Register a student "
            "in the Add New Face tab first."
        )

    st.info(
        "Allow camera access and keep your face visible "
        "for 10 seconds."
    )

    webrtc_streamer(
        key="attendance",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=FaceRecognition(knn).transform,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )

    st.divider()
    st.subheader("📊 Attendance Dashboard")

    path = attendance_file()

    if os.path.exists(path):
        try:
            df = pd.read_csv(path)

            st.metric("Total Attendance Entries", len(df))
            st.dataframe(df, use_container_width=True)

            with open(path, "rb") as f:
                st.download_button(
                    "⬇️ Download Attendance CSV",
                    data=f.read(),
                    file_name=os.path.basename(path),
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Unable to read attendance file: {e}")
    else:
        st.info("No attendance recorded today.")

    if st.button("🔄 Refresh Attendance Dashboard"):
        st.rerun()


# =========================================================
# ADD FACE TAB
# =========================================================

with add_face_tab:
    st.subheader("➕ Register a New Student")

    student_name = st.text_input(
        "Student name",
        placeholder="Enter the student's name",
        key="student_name_input"
    ).strip()

    if student_name:
        st.info(
            "Look at the camera and move your head slightly "
            "to capture varied samples."
        )

        # One persistent callback object per camera key.
        if "add_face_name" not in st.session_state:
            st.session_state.add_face_name = None

        if (
            st.session_state.add_face_name != student_name
            or "add_face_callback" not in st.session_state
        ):
            st.session_state.add_face_name = student_name
            st.session_state.add_face_callback = AddFace(student_name)

        webrtc_streamer(
            key="add_face_camera",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=(
                st.session_state.add_face_callback.transform
            ),
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            async_processing=True
        )

        if st.button("Check Registration Status"):
            callback = st.session_state.add_face_callback

            if callback.saved:
                st.success(
                    f"{student_name} was registered successfully."
                )
                st.session_state.model_version += 1
                st.cache_resource.clear()
                st.rerun()
            elif callback.save_error:
                st.error(callback.save_error)
            else:
                st.info(
                    f"Captured {len(callback.samples)} "
                    f"of {MAX_FACE_SAMPLES} samples. "
                    "Keep the camera running."
                )
    else:
        st.warning("Enter a student name to start registration.")

    st.divider()
    st.subheader("📁 Registered Students")

    if os.path.exists(NAMES_FILE):
        try:
            with open(NAMES_FILE, "rb") as f:
                registered_names = pickle.load(f)

            unique_names = list(dict.fromkeys(registered_names))

            if unique_names:
                st.dataframe(
                    pd.DataFrame({
                        "No.": range(1, len(unique_names) + 1),
                        "Student Name": unique_names
                    }),
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.info("No students registered yet.")

        except Exception as e:
            st.error(f"Unable to load student list: {e}")
    else:
        st.info("No students registered yet.")