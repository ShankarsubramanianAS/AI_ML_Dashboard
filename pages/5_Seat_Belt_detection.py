
# activating the seatbelt detection conda env 
import subprocess
subprocess.call('conda activate seatbelt', shell=True)


import streamlit as st
import time
import cv2
import numpy as np
import tempfile
import os 


video_file = open("static/videos/seatbeltdetect.mp4", 'rb')
video_bytes = video_file.read()

st.subheader("SeatBelt detection")
st.video(video_bytes)
text = st.markdown("")

