import cv2
import os
import time
import streamlit as st
from PIL import Image, GifImagePlugin
#import IPython
#from IPython.display import display, HTML, Image
#from IPython.display import  Image as IMG
#import numpy as np
from tqdm import tqdm 
from scipy.io import wavfile
#import av
import torch
import tempfile
from PIL import Image 
import sys
#sys.path.insert(0, './yolov8')

#st.set_page_config(page_title="AKKODIS AI DASHBOARD", page_icon=":guardsman:", layout="centered")
#header_image = Image.open("C:/Users/shank/Desktop/aiml/ai_dashboard//static/images/header.jpeg")
#st.sidebar.image(header_image,use_column_width=True , output_format="JPEG")
st.set_page_config(page_title="Detection & Segmentation", page_icon="")

#st.sidebar.title("Select App Mode")
app_mode = st.sidebar.selectbox('Choose the App Mode',
                                ['Play Demo','Car Damage Detection'])


# Define functions

def get_camera_stream():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow("Camera Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def load_gif(path):
    return open(path, "rb").read()


# Detection & Segmentation
if app_mode == 'Play Demo':
    st.subheader("Detection & Segmentation")
    seg_gif = load_gif("static/segmentation1.gif")
    st.image(seg_gif, width=680, output_format="GIF")
    st.sidebar.caption("Hyperparameter Tuning")
    st.sidebar.select_slider('Select a Threshold',options=['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7'])
    st.sidebar.select_slider('Select Epochs',options=['20', '40', '60', '80', '100'])
    if st.sidebar.button("Apply"):
        my_bar = st.sidebar.progress(0)
        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1)
        
    # if st.button("Try Detection & Segmentation"):
    #     get_camera_stream()


#from ultralytics import YOLO


#car_file_path ="C:/Users/shank/Desktop/aiml/car_damage"
#weights_file_path = car_file_path + "/weights/8s/best.pt"
#demo_img = car_file_path + "/data/test/76.jpg"
#demo_video = car_file_path + "videoplayback.mp4"

video_file = open("static/videos/cardamagedetect.mp4", 'rb')
video_bytes = video_file.read()

if app_mode == 'Car Damage Detection':
    st.subheader("Detected Damages")
    st.video(video_bytes)
    text = st.markdown("")


    
