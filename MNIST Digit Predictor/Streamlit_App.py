#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 12:28:59 2025

@author: ibrahim
"""

import streamlit as st 
import joblib 
import numpy as np 
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load the model
model = joblib.load('mnist_model.pkl') 

# Streamlit app 
st.title("MNIST Digit Predictor") 

# Choose an option
option = st.sidebar.radio("Choose an option", ["Upload an image", "Draw a digit", "Capture image via webcam"])

# Option 1: Upload an image
if option == "Upload an image":
    uploaded_file = st.file_uploader("Upload an image of a digit", type=["png", "jpg", "jpeg"]) 
    if uploaded_file is not None: 
        image = Image.open(uploaded_file).convert('L')  # Convert to grayscale 
        image = image.resize((28, 28))  # Resize to 28x28 
        image_array = np.array(image).reshape(1, -1)  # Flatten to 1D array 
        if st.button("Predict"):
            prediction = model.predict(image_array) 
            st.write(f"Predicted digit: {prediction[0]}") 
           

# Option 2: Draw a digit 
elif option == "Draw a digit":
    st.write("Draw a digit below:") 
    canvas_result = st_canvas(
        fill_color="black", 
        stroke_width=10, 
        stroke_color="white", 
        background_color="black", 
        width=280, 
        height=280, 
        drawing_mode="freedraw"
    ) 
    if canvas_result.image_data is not None: 
        image = Image.fromarray(canvas_result.image_data).convert('L')  # Convert to grayscale
        image = image.resize((28, 28))  # Resize to 28x28
        image_array = np.array(image).reshape(1, -1)  # Flatten to 1D array
        if st.button("Predict"):
            prediction = model.predict(image_array)
            st.write(f"Predicted digit: {prediction[0]}")
            st.write(model)

# Option 3: Capture image via webcam
elif option == "Capture image via webcam":
    img_file_buffer = st.camera_input("Capture an image", key="camera")
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer).convert('L')  # Convert to grayscale
        image = image.resize((28, 28))  # Resize to 28x28
        image_array = np.array(image).reshape(1, -1)  # Flatten to 1D array
        if st.button("Predict"):
            prediction = model.predict(image_array) 
            st.write(f"Predicted digit: {prediction[0]}")