import cv2
import os
import time
import streamlit as st
from PIL import Image, GifImagePlugin
import sounddevice as sd
import IPython
from IPython.display import display, HTML
import numpy as np
from tqdm import tqdm 
import scipy.io.wavfile as wavfile
from scipy.io import wavfile
import noisereduce as nr
import av
import torch
import tempfile


## Assign file paths here
audio_file_path = "C:/Users/shank/Desktop/aiml/ai_dashboard/static/audio/audio.wav"



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

st.set_page_config(page_title="Driving Assistance", page_icon="🚗")


# Driving Assistance

#if page == "Driving Assistance":
st.subheader("Driving Assistance")
st.image("https://shaoanlu.files.wordpress.com/2017/05/ezgif-2-8a23e30507.gif", width=680, output_format="GIF")
if st.button("Try Driving Assistance"):
    get_camera_stream()

