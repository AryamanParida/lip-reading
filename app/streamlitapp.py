# Importing dependencies
import streamlit as st
import os 
import imageio 
import numpy as np
import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('This application developed by me.')

st.title('LipNet Full Stack App') 
# Generating a list of options or videos 
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

# Generate two columns 
col1, col2 = st.columns(2)

if options: 
    # selected_video_path = os.path.join('..', 'data', 's1', selected_video)
    selected_video_path= os.path.join('..', 'data', 's1', selected_video)
    # Rendering the video 
    with col1: 
        
        st.info('The video below displays the converted video in mp4 format')
        os.system(f'ffmpeg -i {selected_video_path} -vcodec libx264 test_video.mp4 -y')
        # //converting to  mp4
        

        # video = None
        
        # Rendering inside of the app
        video = open('test_video.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)

    with col2: 
        st.info('This is all the machine learning model sees when making a prediction')
        video, annotations = load_data(tf.convert_to_tensor(selected_video_path))
        video = np.array(video)
        video_pixels = (video.astype(np.uint8) * 255).squeeze()
        imageio.mimsave('animation.gif', video_pixels, duration=50)
        st.image('animation.gif', width=400) 

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()

        # st.text(decoder)
        st.info('Real text')
        # converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(tf.strings.reduce_join(num_to_char(annotations)).numpy().decode('utf-8'))

        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)