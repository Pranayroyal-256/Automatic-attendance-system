import streamlit as st
import pandas as pd
import os
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Attendance Dashboard")

st.title("Automatic Attendance System Dashboard")

st_autorefresh(interval=5000, key="refresh")

try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    attendance_folder = os.path.join(BASE_DIR, "attendance")

    st.write("Attendance folder:")
    st.write(attendance_folder)

    if os.path.exists(attendance_folder):

        files = os.listdir(attendance_folder)

        csv_files = [f for f in files if f.endswith(".csv")]

        if csv_files:
            latest_file = sorted(csv_files)[-1]

            file_path = os.path.join(attendance_folder, latest_file)

            df = pd.read_csv(file_path)

            st.success("Attendance Loaded Successfully")
            st.dataframe(df)

        else:
            st.warning("No CSV attendance files found")

    else:
        st.error("Attendance folder not found")

except Exception as e:
    st.error("Application Error:")
    st.exception(e)