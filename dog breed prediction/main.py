import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

#loading the dataset
model = load_model('dog_breed.h5')

#Name of the classess
CLASS_NAMES = ['Scottish Deerhound','Maltese Dog','Bernese Mountain Dog']

#setting title of app

st.title("Dog Breed Predictor")
st.markdown("upload an image of the dog")

#uploading the dog image
dog_image = st.file_uploader("Choose an image....", type = (["jpg","png"]))
submit = st.button('Predict')

if submit:

    if dog_image is not None:
        file_bytes = np.asarray(bytearray(dog_image.read()),dtype = np.uint8)
        opencv_image = cv2.imdecode(file_bytes,1)

        st.image(opencv_image,channels="BGR")

        opencv_image = cv2.resize(opencv_image,(224,224))
        opencv_image.shape = (1,224,224,3)
        Y_pred = model.predict(opencv_image)
        

        result = st.title(str("The Dog Breed is " + CLASS_NAMES[np.argmax(Y_pred)]))
        st.title(result)

    else:
        st.title(str("Please upload an image"))



