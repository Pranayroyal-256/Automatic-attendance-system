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


# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Automatic Attendance System",
    page_icon="🎓",
    layout="wide"
)


# ---------------- VARIABLES ----------------

THRESHOLD_TIME = 60

marked_names = set()
detection_times = {}


# ---------------- LOAD FACE MODEL ----------------


@st.cache_resource
def load_model():

    with open("data/names.pkl", "rb") as f:
        LABELS = pickle.load(f)

    with open("data/faces_data.pkl", "rb") as f:
        FACES = pickle.load(f)


    size = min(len(FACES), len(LABELS))

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



knn = load_model()



# ---------------- FACE DETECTOR ----------------


facedetect = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)



# ---------------- ATTENDANCE FILE ----------------


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




# ---------------- MARK ATTENDANCE ----------------


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



# ---------------- WEBRTC CAMERA ----------------


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



        for (x,y,w,h) in faces:


            crop = img[
                y:y+h,
                x:x+w
            ]


            try:

                resized = cv2.resize(
                    crop,
                    (50,50)
                )


                resized = resized.flatten()

                resized = resized.reshape(
                    1,-1
                )


                name = str(
                    knn.predict(resized)[0]
                )



                cv2.rectangle(
                    img,
                    (x,y),
                    (x+w,y+h),
                    (0,255,0),
                    2
                )



                cv2.putText(
                    img,
                    name,
                    (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,255,255),
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
                        (x,y+h+40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,255,0),
                        2
                    )



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


                    cv2.putText(
                        img,
                        "Attendance Marked",
                        (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,0),
                        3
                    )



            except Exception as e:

                print(e)



        return img





# ---------------- UI ----------------



st.title(
    "🎓 Automatic Attendance System"
)


st.write(
    "Face Recognition Based Attendance System"
)



st.info(
    "Allow camera permission and stay in front of camera for 60 seconds."
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




# ---------------- DASHBOARD ----------------


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
        "Total Students",
        len(df)
    )


else:


    st.warning(
        "No attendance recorded today"
    )