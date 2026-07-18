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


# ---------------- CONFIG ----------------

st.set_page_config(
    page_title="Automatic Attendance System",
    page_icon="🎓",
    layout="wide"
)


THRESHOLD_TIME = 60

marked_names = set()
detection_times = {}


# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():

    with open("data/names.pkl", "rb") as f:
        LABELS = pickle.load(f)

    with open("data/faces_data.pkl", "rb") as f:
        FACES = pickle.load(f)


    min_len = min(len(FACES), len(LABELS))

    FACES = FACES[:min_len]
    LABELS = LABELS[:min_len]


    knn = KNeighborsClassifier(
        n_neighbors=5
    )

    knn.fit(
        FACES,
        LABELS
    )

    return knn



knn = load_model()



facedetect = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)



# ---------------- ATTENDANCE FILE ----------------


def get_attendance_file():

    today = datetime.now().strftime("%d-%m-%Y")

    folder = "attendance"


    if not os.path.exists(folder):
        os.makedirs(folder)


    file = f"{folder}/Attendance_{today}.csv"


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




# ---------------- SAVE ATTENDANCE ----------------


def mark_attendance(name):

    file = get_attendance_file()

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


# ---------------- UI ----------------


st.title(
    "🎓 Automatic Attendance System"
)


st.write(
    "Face Recognition Based Attendance System"
)



start = st.button(
    "📷 Start Attendance"
)



FRAME_WINDOW = st.empty()



# ---------------- CAMERA ----------------


if start:


    camera = cv2.VideoCapture(0)


    if not camera.isOpened():

        st.error(
            "Camera not detected"
        )

        st.stop()



    while True:


        ret, frame = camera.read()


        if not ret:
            st.error(
                "Cannot read camera"
            )
            break



        frame = cv2.resize(
            frame,
            (640,480)
        )



        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )



        faces = facedetect.detectMultiScale(
            gray,
            1.3,
            5
        )



        for (x,y,w,h) in faces:


            crop = frame[
                y:y+h,
                x:x+w
            ]


            try:

                resized = cv2.resize(
                    crop,
                    (50,50)
                ).flatten().reshape(1,-1)


                name = str(
                    knn.predict(resized)[0]
                )


            except:

                continue



            cv2.rectangle(
                frame,
                (x,y),
                (x+w,y+h),
                (0,255,0),
                2
            )


            cv2.putText(
                frame,
                name,
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,255,255),
                2
            )



            # -------- TIMER --------


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
                    frame,
                    f"Stay {remaining}s",
                    (x,y+h+40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2
                )



            # -------- MARK ATTENDANCE --------


            if (
                elapsed >= THRESHOLD_TIME
                and name not in marked_names
            ):


                mark_attendance(
                    name
                )


                marked_names.add(
                    name
                )


                st.success(
                    f"✅ Attendance marked for {name}"
                )



        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        FRAME_WINDOW.image(
            frame
        )



    camera.release()




# ---------------- DASHBOARD ----------------


st.subheader(
    "📊 Attendance Dashboard"
)



file = get_attendance_file()



if os.path.exists(file):


    df = pd.read_csv(file)


    st.dataframe(
        df,
        use_container_width=True
    )


    st.metric(
        "Total Students",
        len(df)
    )



else:

    st.info(
        "No attendance today"
    )