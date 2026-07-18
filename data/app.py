import streamlit as st
import pandas as pd
import os
import time
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

st.title("Automatic Attendance System Dashboard")

# Auto refresh
count = st_autorefresh(interval=5000, key="refresh")

today = datetime.now().strftime("%d-%m-%Y")

st.write("Today's Date:", today)

attendance_folder = "attendance"

if os.path.exists(attendance_folder):
    files = os.listdir(attendance_folder)
else:
    files = []

csv_files = [file for file in files if file.endswith(".csv")]

if csv_files:
    latest_file = sorted(csv_files)[-1]

    path = os.path.join(attendance_folder, latest_file)

    df = pd.read_csv(path)

    st.success("Attendance Loaded Successfully")

    st.dataframe(df)

else:
    st.warning("No attendance records found yet.")