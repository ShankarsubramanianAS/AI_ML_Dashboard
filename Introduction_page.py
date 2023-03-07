import streamlit as st
from PIL import Image

st.set_page_config(page_title="AKKODIS AI DASHBOARD", page_icon=":guardsman:", layout="centered")
header_image = Image.open("AI_ML_Dashboard/static/images/header.jpeg")
st.image(header_image,use_column_width=True , output_format="JPEG")

## Assign file paths here
audio_file_path = "AI_ML_Dashboard/static/audio/audio.wav"

st.markdown(
    """
    List of DEMO's \n
    1. Detection & Segmentation 
    2. Driving Assistance
    3. Rearview Assistance
    4. Fire Detection
    5. Seat Belt Detection
    6. AI Noise Cancellation
    7. NLP with GPT3
Please access the demos through the side menu
     """)



# page = st.sidebar.selectbox("Select an AI Solution", [
#   "Detection & Segmentation",
#   "Driving Assistance",
#   "Rearview Assistance",
#   "Fire Detection",
#   "Seat Belt Detection",
#   "AI Noise Cancellation"
# ])

# Define functions

# def get_camera_stream():
#     cap = cv2.VideoCapture(0)
#     while True:
#         ret, frame = cap.read()
#         cv2.imshow("Camera Stream", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()

# def load_gif(path):
#     return open(path, "rb").read()

# def record_audio(filename, duration):
#     fs = 44100  # sample rate
#     recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
#     sd.wait()  # Wait until recording is finished
#     wavfile.write(filename, fs, recording)  # Save recording to a .wav file

# def play_audio(filename):
#     fs, recording = wavfile.read(filename)  # Load recording from .wav file
#     sd.play(recording, blocking=True)  # Play recording

# # Detection & Segmentation

# if page == "Detection & Segmentation":
#     st.subheader("Detection & Segmentation")
#     seg_gif = load_gif("C:/Users/shank/Desktop/aiml/ai_dashboard/static/segmentation1.gif")
#     st.image(seg_gif, width=680, output_format="GIF")
#     st.sidebar.caption("Hyperparameter Tuning")
#     st.sidebar.select_slider('Select a Threshold',options=['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7'])
#     st.sidebar.select_slider('Select Epochs',options=['20', '40', '60', '80', '100'])
#     if st.sidebar.button("Apply"):
#         my_bar = st.sidebar.progress(0)
#         for percent_complete in range(100):
#             time.sleep(0.1)
#             my_bar.progress(percent_complete + 1)
    
#     if st.button("Try Detection & Segmentation"):
#         get_camera_stream()

# # Driving Assistance

# elif page == "Driving Assistance":
#     st.subheader("Driving Assistance")
#     st.image("https://shaoanlu.files.wordpress.com/2017/05/ezgif-2-8a23e30507.gif", width=680, output_format="GIF")
#     if st.button("Try Driving Assistance"):
#         get_camera_stream()

# # Rearview Assistance

# elif page == "Rearview Assistance":
#     st.subheader("Rearview Assistance")
#     st.image("https://s3-ap-northeast-1.amazonaws.com/dod-tec/files/BSD300/BSD300_02.gif", width=680, output_format="GIF")
#     if st.button("Try Rearview Assistance"):
#         get_camera_stream()


# # Fire Detection

# elif page == "Fire Detection":
#     st.subheader("Rearview Assistance")
#     st.image("https://s3-ap-northeast-1.amazonaws.com/dod-tec/files/BSD300/BSD300_02.gif", width=680, output_format="GIF")
#     if st.button("Try Rearview Assistance"):
#         get_camera_stream()

# # Seat Belt Detection

# elif page == "Seat Belt Detection":
#     st.subheader("Seat Belt Detection")
#     st.image(Image.open("C:/Users/shank/Desktop/aiml/ai_dashboard/static/images/seatbelt.jpg"), width=680, output_format="JPEG")
#     if st.button("Try Seat Belt Detection"):
#         get_camera_stream()

# # AI Noise Cancellation

# elif page == "AI Noise Cancellation":
#     st.subheader("AI Noise Cancellation")
#     st.image(Image.open("C:/Users/shank/Desktop/aiml/ai_dashboard/static/images/audio.jpg"), width=680, output_format="JPEG")
#     if st.sidebar.button("Record Audio"):
#         audio_file = audio_file_path
#         if os.path.exists(audio_file):
#             os.remove(audio_file)
#         duration = 15  # recording duration in seconds
#         #st.write("Recording in progress...")

#         ph = st.empty()
#         my_bar = st.progress(0)
#         for secs in range(duration, 0, -1):
#             mm, ss = secs//60, secs%60
#             my_bar.progress(secs + 1)
#             ph.metric("Recording.. ", f"{mm:02d}:{ss:02d}")
#             time.sleep(1)

#         record_audio(audio_file, duration)
#         #st.write("Recording complete.")
#         # load data
#         rate, data = wavfile.read(audio_file)
#         #IPython.display.Audio(data=data, rate=rate)
#         # perform noise reduction
#         reduced_noise = nr.reduce_noise(y=data, sr=rate)
#         new_audio = wavfile.write("mywav_reduced_noise.wav", rate, reduced_noise)
    
#     if st.sidebar.button("Play Audio"):
#         #audio_file = "mywav_reduced_noise.wav"
#         play_audio(new_audio)
#         #st.write("Playing audio.")



