import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained model
@st.cache_resource()  # Cache the model to avoid reloading on every run
def load_pretrained_model(model_path):
    return load_model(model_path)

# Function to preprocess the uploaded image
def preprocess_image(uploaded_image):
    img = image.load_img(uploaded_image, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions
def predict_image(model, img_array):
    predictions = model.predict(img_array)
    return np.argmax(predictions)

# Title for the app
st.title("Lung Disease Detector")
st.write("This application detects whether a lung is normal or is infected based on the X ray image given as input")

# File uploader widget
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

class_val_map={0:'covid_infected',1:'Lung opacity',2:'Normal',3:'Viral pneumonia'}

st.write(" ")
st.write(" ")

# Check if an image was uploaded
if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Load the pre-trained model
    model = load_pretrained_model("model.h5")

    # Preprocess the uploaded image
    img_array = preprocess_image(uploaded_file)

    # Make predictions
    predicted_class = predict_image(model, img_array)

    # Display the predicted class
    st.write("The predicted disease is:", class_val_map[predicted_class])
