import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
from utils.process_static_sign import process_static_sign
from utils.process_dynamic_sign import process_dynamic_sign

def static_video_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    imgOutput = process_static_sign(img)

    return av.VideoFrame.from_ndarray(imgOutput, format="bgr24")

def dynamic_video_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    imgOutput = process_dynamic_sign(img)

    return av.VideoFrame.from_ndarray(imgOutput, format="bgr24")

def main():
    st.title("Hand Gesture Classifier")
    st.write("This app detects and classifies hand gestures in real-time.")

    tab1, tab2 = st.tabs(["Static Sign Detection", "Dynamic Sign Detection"])

    with tab1:
        st.header("Static Sign Detection")
        webrtc_streamer(key="static", video_frame_callback=static_video_callback)

    with tab2:
        st.header("Dynamic Sign Detection")
        webrtc_streamer(key="dynamic", video_frame_callback=dynamic_video_callback)
        
if __name__ == "__main__":
    main()
