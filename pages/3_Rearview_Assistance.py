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

## Assign file paths here
audio_file_path = "static/audio/audio.wav"



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

def record_audio(filename, duration):
    fs = 44100  # sample rate
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    wavfile.write(filename, fs, recording)  # Save recording to a .wav file

def play_audio(filename):
    fs, recording = wavfile.read(filename)  # Load recording from .wav file
    sd.play(recording, blocking=True)  # Play recording

st.set_page_config(page_title="Rearview Assistance", page_icon="🚗")


# Rearview Assistance

st.subheader("Rearview Assistance")
st.image("https://s3-ap-northeast-1.amazonaws.com/dod-tec/files/BSD300/BSD300_02.gif", width=680, output_format="GIF")
if st.button("Try Rearview Assistance"):
    get_camera_stream()

