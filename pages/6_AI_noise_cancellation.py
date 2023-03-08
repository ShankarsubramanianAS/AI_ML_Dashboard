# activating the seatbelt detection conda env 
import cv2
import os
import time
import streamlit as st
from PIL import Image, GifImagePlugin
#import IPython
#from IPython.display import display, HTML, Image
#from IPython.display import  Image as IMG
import numpy as np
from tqdm import tqdm 
from scipy.io import wavfile
#import av
import torch
import tempfile
from PIL import Image 
import sys

import streamlit as st
import torch
import torchaudio
from PIL import Image, ImageOps
from denoiser import pretrained
from denoiser.dsp import convert_audio
import sounddevice as sd
import wave
import os
import moviepy.editor as mp
import cv2
from pydub import AudioSegment
from io import BytesIO
from audio_recorder_streamlit import audio_recorder
from audiorecorder import audiorecorder
import  streamlit_toggle as tog
import time



import psutil


st.set_page_config(page_title="Ai Noise cancellation", page_icon="")

#st.sidebar.title("Select App Mode")
app_mode = st.sidebar.selectbox('Choose the App Mode',
                                ['Play Demo','Run on Audio','Record your own audio'])

video = cv2.VideoCapture('C:/Users/shank/Desktop/aiml/ai_dashboard/static/audio/boeingtest.mp4')
video_file = open("C:/Users/shank/Desktop/aiml/ai_dashboard/static/audio/boeingtest.mp4", 'rb')
video_bytes = video_file.read()
audio_file = "C:/Users/shank/Desktop/aiml/ai_dashboard/static/audio/boeingtest.mp3"


# Define a function to play and pause the audio
def play_audio(audio_file):
    # Use streamlit.audio to play the audio file
    audio = st.audio(audio_file, format="audio/mp3", start_time=0)


if app_mode == 'Play Demo':

    s=0

    st.title("AI Noise Cancellation for Audio")

    st.markdown("Original Audio before AI Noise Reduction")

    st.video(video_bytes)

    tog_swicth = tog.st_toggle_switch( 
                    label="  Apply AI Noise cancellation",
                    key="key1", 
                    default_value=False, 
                    label_after = True, 
                    inactive_color = '#D3D3D3', 
                    active_color="#e8b62c", 
                    track_color="#29B5E8"
                    )

    if tog_swicth:

        with st.spinner('Wait for it...'):
            time.sleep(7)
        st.success('Done!')
        s=1

    if s==1:

        st.markdown("Audio After AI Noise Reduction")

        play_audio(audio_file)

            


def denoise_audio(file_path):
    # Load the original audio
    # Denoise the audio and save it to a new file
    model = pretrained.dns64().cuda()
    wav, sr = torchaudio.load(file_path)
    wav = convert_audio(wav.cuda(), sr, model.sample_rate, model.chin)
    with torch.no_grad():
        denoised = model(wav.cuda())[0]
        #denoised = model(wav[None])[0]
    denoised_file_path = os.path.join(folder_path, "denoised_" + uploaded_file.name)
    torchaudio.save(denoised_file_path, denoised.cpu(), model.sample_rate)
    return denoised_file_path

def denoise_audio1(file_path):
    # Load the original audio
    # Denoise the audio and save it to a new file
    model = pretrained.dns64().cuda()
    wav, sr = torchaudio.load(file_path)
    wav = convert_audio(wav.cuda(), sr, model.sample_rate, model.chin)
    with torch.no_grad():
        denoised = model(wav.cuda())[0]
        #denoised = model(wav[None])[0]
    denoised_file_path = os.path.join(folder_path, "denoised_" + "sample.wav")
    torchaudio.save(denoised_file_path, denoised.cpu(), model.sample_rate)
    return denoised_file_path

if app_mode == 'Run on Audio':

    # Set the title and subtitle of the app
    #st.title("Audio Denoiser")
    st.subheader("Upload an audio file in MP3 format to denoise it with AI")

    # Get the uploaded file from the user and save it to a folder
    uploaded_file = st.file_uploader("Choose an MP3 file", type="mp3")

    if uploaded_file is not None:
        # Save the file to a folder named "uploads"
        folder_path = "C:/Users/shank/Desktop/aiml/ai_dashboard/static/audio/uploads"
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        file_path = os.path.join(folder_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File saved successfully")

        # Set the Markdown text to display the original audio
        st.markdown("### Original Audio")
        st.audio(file_path)

        # Add a "Start Denoising" button
        if st.button("Start Denoising"):
            # Leave space for the denoise code
            st.markdown("---")
            st.markdown(" AI Denoised Audio")

        # Denoise the audio and save it to a new file
            denoised_file_path = denoise_audio(file_path)

            # Set the Markdown text to display the denoised audio
            st.markdown("### Audio after AI Denoising")
            st.audio(denoised_file_path)







if app_mode == 'Record your own audio':

    # Set the title and subtitle of the app
    #st.title("Audio Denoiser")
    st.subheader("Record your voice and try AI denoiser")

    # Records 3 seconds in any case
    audio_bytes = audio_recorder(text="click to record/stop",
    recording_color="#6aa36f",
    neutral_color="#e8b62c",energy_threshold=(-1.0, 1.0),pause_threshold=10.0)

    if audio_bytes:
        st.audio(audio_bytes, format="audio/mp3")

        folder_path = "C:/Users/shank/Desktop/aiml/ai_dashboard/static/audio/uploads/"
        wav_file = open(folder_path+"audio.mp3", "wb")
        wav_file.write(audio_bytes)
        aud_file = folder_path+"audio.mp3"
    #audio = audiorecorder("Click here to record", "Recording...")

    # if len(audio) > 0:
    #     # To play audio in frontend:
    #     st.audio(audio.tobytes())
        
    #     # To save audio to a file:
    #     save_path = "C:/Users/shank/Desktop/aiml/ai_dashboard/static/audio/uploads/"
    #     wav_file = open(save_path+"audio.wav", "wb")
    #     wav_file.write(audio.tobytes())

        # Add a "Start Denoising" button
        if st.button("Start Denoising"):
                # Leave space for the denoise code
            st.markdown("---")
            st.markdown(" AI Denoised Audio")

            # Denoise the audio and save it to a new file
            denoised_audio = denoise_audio1(aud_file)

                # Set the Markdown text to display the denoised audio
            st.markdown("### Audio after AI Denoising")
            st.audio(denoised_audio)



    if st.sidebar.button("Clear / Refresh"):
        # Set the directory and file path
        folder_path = "C:/Users/shank/Desktop/aiml/ai_dashboard/static/audio/uploads/"
        file_path = os.path.join(folder_path, "audio.mp3")

        if os.path.isfile(file_path):
            os.remove(file_path)
            st.success("Refreshed successfully!")
        else:
            st.warning("Good to go!!")
