import streamlit as st
import cv2
import numpy as np
import av
import torch
import tempfile
from PIL import Image

fire_file_path ="C:/Users/shank/Desktop/aiml/fire_detection/Fire_Detection_YoloV5/"
weights_file_path = fire_file_path + "weights/best.pt"

@st.cache()
def load_model():
    model = torch.hub.load('ultralytics/yolov5','custom',path=weights_file_path ,force_reload=True)
    return model

demo_img = fire_file_path + "fire.9.jpg"
demo_video = fire_file_path + "videoplayback.mp4"

st.set_page_config(page_title="Fire Detection", page_icon="🔥")


app_mode = st.sidebar.selectbox('Choose the App Mode',
                                ['Run on Image','Run on Video','Run on WebCam'])


if app_mode == 'Run on Image':
    st.subheader("Detected Fire:")
    text = st.markdown("")
    
    st.sidebar.markdown("---")
    # Input for Image
    img_file = st.sidebar.file_uploader("Upload an Image",type=["jpg","jpeg","png"])
    if img_file:
        image = np.array(Image.open(img_file))
    else:
        image = np.array(Image.open(demo_img))
        
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Original Image**")
    st.sidebar.image(image)
    
    # predict the image
    model = load_model()
    results = model(image)
    length = len(results.xyxy[0])
    if length > 0:
        length1 = "Fire Alert !!!"
    else:
        length1 = "All Clear"
    output = np.squeeze(results.render())
    text.write(f"<h1 style='text-align: center; color:orange;'>{length1}</h1>",unsafe_allow_html = True)
    st.subheader("Output Image")
    st.image(output,use_column_width=True)
    
if app_mode == 'Run on Video':
    st.subheader("Detected Fire:")
    text = st.markdown("")
    
    st.sidebar.markdown("---")
    
    st.subheader("Output")
    stframe = st.empty()
    
    #Input for Video
    video_file = st.sidebar.file_uploader("Upload a Video",type=['mp4','mov','avi','asf','m4v'])
    st.sidebar.markdown("---")
    tffile = tempfile.NamedTemporaryFile(delete=False)
    
    if not video_file:
        vid = cv2.VideoCapture(demo_video)
        tffile.name = demo_video
    else:
        tffile.write(video_file.read())
        vid = cv2.VideoCapture(tffile.name)
    
    st.sidebar.markdown("**Input Video**")
    st.sidebar.video(tffile.name)
    
    # predict the video
    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        model = load_model()
        results = model(frame)
        length = len(results.xyxy[0])
        if length > 0:
            length1 = "Fire Alert !!!"
        else:
            length1 = "All Clear"
        output = np.squeeze(results.render())
        text.write(f"<h1 style='text-align: center; color:orange;'>{length1}</h1>",unsafe_allow_html = True)
        stframe.image(output)
        
if app_mode == 'Run on WebCam':
    st.subheader("Detected Fire:")
    text = st.markdown("")
    
    st.sidebar.markdown("---")
    
    st.subheader("Output")
    stframe = st.empty()
    
    run = st.sidebar.button("Start")
    stop = st.sidebar.button("Stop")
    st.sidebar.markdown("---")
    
    cam = cv2.VideoCapture(0)
    if(run):
        while(True):
            if(stop):
                break
            ret,frame = cam.read()
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            model = load_model()
            results = model(frame)
            length = len(results.xyxy[0])
            if length > 0:
                length1 = "Fire Alert !!!"
            else:
                length1 = "All Clear"
            output = np.squeeze(results.render())
            text.write(f"<h1 style='text-align: center; color:orange;'>{length1}</h1>",unsafe_allow_html = True)
            stframe.image(output)
