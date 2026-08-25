import streamlit as st
import cv2
import pickle
import numpy as np
import os
import csv
import time
import pandas as pd

from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier

from streamlit_webrtc import (
    webrtc_streamer,
    VideoTransformerBase,
    RTCConfiguration
)


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Automatic Attendance System",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# VARIABLES
# =========================================================

THRESHOLD_TIME = 60

marked_names = set()
detection_times = {}


# =========================================================
# CREATE REQUIRED FOLDERS
# =========================================================

os.makedirs("data", exist_ok=True)
os.makedirs("attendance", exist_ok=True)


# =========================================================
# FACE DETECTOR
# =========================================================

facedetect = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

if facedetect.empty():
    st.error(
        "❌ haarcascade_frontalface_default.xml not found."
    )
    st.stop()


# =========================================================
# LOAD FACE MODEL
# =========================================================

def get_file_version():

    faces_file = "data/faces_data.pkl"
    names_file = "data/names.pkl"

    faces_time = (
        os.path.getmtime(faces_file)
        if os.path.exists(faces_file)
        else 0
    )

    names_time = (
        os.path.getmtime(names_file)
        if os.path.exists(names_file)
        else 0
    )

    return faces_time, names_time


@st.cache_resource
def load_model(faces_version, names_version):

    with open(
        "data/names.pkl",
        "rb"
    ) as f:

        LABELS = pickle.load(f)


    with open(
        "data/faces_data.pkl",
        "rb"
    ) as f:

        FACES = pickle.load(f)


    FACES = np.asarray(FACES)


    size = min(
        len(FACES),
        len(LABELS)
    )


    FACES = FACES[:size]
    LABELS = LABELS[:size]


    knn = KNeighborsClassifier(
        n_neighbors=5
    )


    knn.fit(
        FACES,
        LABELS
    )


    return knn


# Load model
faces_version, names_version = get_file_version()

knn = load_model(
    faces_version,
    names_version
)


# =========================================================
# ATTENDANCE FILE
# =========================================================

def attendance_file():

    today = datetime.now().strftime(
        "%d-%m-%Y"
    )


    folder = "attendance"


    if not os.path.exists(folder):
        os.makedirs(folder)


    file = (
        f"{folder}/Attendance_{today}.csv"
    )


    if not os.path.exists(file):

        with open(
            file,
            "w",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                [
                    "NAME",
                    "TIME"
                ]
            )


    return file


# =========================================================
# MARK ATTENDANCE
# =========================================================

def mark_attendance(name):

    file = attendance_file()


    timestamp = datetime.now().strftime(
        "%H:%M:%S"
    )


    with open(
        file,
        "a",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                name,
                timestamp
            ]
        )


# =========================================================
# WEBRTC CONFIGURATION
# =========================================================

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers":
        [
            {
                "urls":
                [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    }
)


# =========================================================
# ATTENDANCE FACE RECOGNITION
# =========================================================

class FaceRecognition(VideoTransformerBase):

    def transform(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )


        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )


        faces = facedetect.detectMultiScale(
            gray,
            1.3,
            5
        )


        for (x, y, w, h) in faces:

            crop = img[
                y:y+h,
                x:x+w
            ]


            try:

                resized = cv2.resize(
                    crop,
                    (50, 50)
                )


                resized = resized.flatten()


                resized = resized.reshape(
                    1, -1
                )


                name = str(
                    knn.predict(resized)[0]
                )


                # Face rectangle
                cv2.rectangle(
                    img,
                    (x, y),
                    (x+w, y+h),
                    (0, 255, 0),
                    2
                )


                # Name
                cv2.putText(
                    img,
                    name,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )


                current_time = time.time()


                if name not in detection_times:

                    detection_times[name] = current_time


                elapsed = (
                    current_time -
                    detection_times[name]
                )


                remaining = int(
                    THRESHOLD_TIME -
                    elapsed
                )


                if remaining > 0:

                    cv2.putText(
                        img,
                        f"Stay {remaining}s",
                        (x, y+h+40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )


                if (
                    elapsed >= THRESHOLD_TIME
                    and name not in marked_names
                ):

                    mark_attendance(name)

                    marked_names.add(name)

                    cv2.putText(
                        img,
                        "Attendance Marked",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        3
                    )


            except Exception as e:

                print(e)


        return img


# =========================================================
# ADD FACE TRANSFORMER
# =========================================================

class AddFace(VideoTransformerBase):

    def __init__(self, student_name):

        self.student_name = student_name

        self.faces_data = []

        self.frame_count = 0

        self.saved = False


    def transform(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )


        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )


        faces = facedetect.detectMultiScale(
            gray,
            1.3,
            5
        )


        for (x, y, w, h) in faces:

            crop_img = img[
                y:y+h,
                x:x+w
            ]


            try:

                resized_img = cv2.resize(
                    crop_img,
                    (50, 50)
                )


                self.frame_count += 1


                # Capture every 5th frame
                if (
                    self.frame_count % 5 == 0
                    and len(self.faces_data) < 100
                ):

                    self.faces_data.append(
                        resized_img
                    )


                # Face rectangle
                cv2.rectangle(
                    img,
                    (x, y),
                    (x+w, y+h),
                    (0, 255, 0),
                    2
                )


                # Student name
                cv2.putText(
                    img,
                    self.student_name,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )


                # Counter
                cv2.putText(
                    img,
                    f"Samples: {len(self.faces_data)}/100",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2
                )


            except Exception as e:

                print(e)


        # =================================================
        # SAVE AFTER 100 SAMPLES
        # =================================================

        if (
            len(self.faces_data) >= 100
            and not self.saved
        ):

            self.save_data()

            self.saved = True


            cv2.putText(
                img,
                "FACE ADDED SUCCESSFULLY!",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


        return img


    # =====================================================
    # SAVE FACE DATA
    # =====================================================

    def save_data(self):

        faces_data = np.asarray(
            self.faces_data
        )


        faces_data = faces_data.reshape(
            100,
            -1
        )


        # -----------------------------------------------
        # SAVE NAMES
        # -----------------------------------------------

        names_file = "data/names.pkl"


        if os.path.exists(names_file):

            with open(
                names_file,
                "rb"
            ) as f:

                names = pickle.load(f)

        else:

            names = []


        names = names + (
            [self.student_name] * 100
        )


        with open(
            names_file,
            "wb"
        ) as f:

            pickle.dump(
                names,
                f
            )


        # -----------------------------------------------
        # SAVE FACES
        # -----------------------------------------------

        faces_file = "data/faces_data.pkl"


        if os.path.exists(faces_file):

            with open(
                faces_file,
                "rb"
            ) as f:

                existing_faces = pickle.load(f)


            existing_faces = np.asarray(
                existing_faces
            )


            faces_data = np.append(
                existing_faces,
                faces_data,
                axis=0
            )


        with open(
            faces_file,
            "wb"
        ) as f:

            pickle.dump(
                faces_data,
                f
            )


# =========================================================
# MAIN UI
# =========================================================

st.title(
    "🎓 Automatic Attendance System"
)


st.write(
    "Face Recognition Based Attendance System"
)


# =========================================================
# TABS
# =========================================================

attendance_tab, add_face_tab = st.tabs(
    [
        "🎥 Attendance",
        "➕ Add New Face"
    ]
)


# =========================================================
# ATTENDANCE TAB
# =========================================================

with attendance_tab:

    st.info(
        "Allow camera permission and stay in front "
        "of the camera for 60 seconds."
    )


    webrtc_streamer(

        key="attendance",

        video_transformer_factory=FaceRecognition,

        rtc_configuration=RTC_CONFIGURATION,

        media_stream_constraints={
            "video": True,
            "audio": False
        }

    )


    # =====================================================
    # DASHBOARD
    # =====================================================

    st.subheader(
        "📊 Attendance Dashboard"
    )


    today = datetime.now().strftime(
        "%d-%m-%Y"
    )


    file = (
        f"attendance/Attendance_{today}.csv"
    )


    if os.path.exists(file):

        df = pd.read_csv(file)


        st.success(
            "Attendance Loaded Successfully"
        )


        st.dataframe(
            df,
            use_container_width=True
        )


        st.metric(
            "Total Attendance",
            len(df)
        )


    else:

        st.warning(
            "No attendance recorded today"
        )


# =========================================================
# ADD FACE TAB
# =========================================================

with add_face_tab:

    st.header(
        "➕ Add New Student Face"
    )


    st.write(
        "Add a new student directly from the camera."
    )


    student_name = st.text_input(
        "👤 Enter Student Name",
        placeholder="Enter student name"
    )


    if student_name.strip() == "":

        st.warning(
            "Please enter the student's name first."
        )

    else:

        st.success(
            f"Ready to register: **{student_name}**"
        )


        st.info(
            "Look directly at the camera. "
            "The system will automatically capture "
            "100 face samples."
        )


        webrtc_streamer(

            key="add_face_camera",

            video_transformer_factory=lambda:
                AddFace(student_name),

            rtc_configuration=RTC_CONFIGURATION,

            media_stream_constraints={
                "video": True,
                "audio": False
            }

        )


    st.markdown("---")


    st.subheader(
        "📁 Registered Students"
    )


    names_file = "data/names.pkl"


    if os.path.exists(names_file):

        with open(
            names_file,
            "rb"
        ) as f:

            registered_names = pickle.load(f)


        unique_names = list(
            dict.fromkeys(
                registered_names
            )
        )


        if len(unique_names) > 0:

            for index, student in enumerate(
                unique_names,
                start=1
            ):

                st.write(
                    f"**{index}.** {student}"
                )

        else:

            st.info(
                "No students registered yet."
            )

    else:

        st.info(
            "No students registered yet."
        )