import streamlit as st
import tensorflow as tf
import keras
import numpy as np
from matplotlib.image import imread
import colorsys
import cv2

st.markdown(
    """
    <style>
    /* Set the background image */
    .stApp {
        background-image: url('https://img.freepik.com/free-photo/flat-lay-beer-bottles-with-chips-nuts_23-2148754981.jpg?t=st=1737990572~exp=1737994172~hmac=e8b153437b6ef164d138d6caac73be5d1776fa91bd4d9e005e50eb6115f3bc3b&w=1060');
        background-size: cover;
        background-position: center;

    .custom-uploader {
        background-color: #FDC41B;
        color: white;
        border-radius: 12px;
        padding: 10px 20px;
        text-align: center;
        font-weight: bold;
        font-size: 18px;
        cursor: pointer;
    }
    
    .custom-uploader:hover {
        background-color: #F1A825;
        color: black;
    
    }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load your trained model (use the appropriate format, .h5 or .keras)
model = tf.keras.models.load_model('C:/Users/divya/Desktop/Machine_Learning_Sebestian/Potato_Chip_CNN/model.keras')  # or 'model_name.h5'

# Function to preprocess the uploaded image
def preprocess_image(uploaded_image):
    img = keras.preprocessing.image.load_img(uploaded_image, target_size=(256, 256))  # Change size to match your model input
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize image if your model was trained with normalized data
    return img_array

# Function to classify the image
def classify_image(img_array):
    prediction = model.predict(img_array)
    return prediction

def segment_image(uploaded_image):
    input_image = imread(uploaded_image)
    input_image_normalized = input_image/255
    R,G,B = input_image_normalized[:,:,0],input_image_normalized[:,:,1],input_image_normalized[:,:,2]
    H, S, V = np.vectorize(lambda r, g, b: colorsys.rgb_to_hsv(r, g, b))(R,G,B)
    V_normalized = np.uint8((255*V))
    
    # Convert to LAB color space
    lab_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2LAB)

    # Threshold the L channel to detect darker areas
    l_channel = lab_image[:, :, 0]
    _, full_chip_mask = cv2.threshold(l_channel, 190, 255, cv2.THRESH_BINARY_INV)  # Shadows have low L values

    kernel1 = np.ones((50,50), np.uint8)  # 5x5 kernel for closing
    closed1 = cv2.morphologyEx(full_chip_mask, cv2.MORPH_CLOSE, kernel1)

    total_pixels = np.sum(full_chip_mask==255)

    # Threshold the L channel to detect darker areas
    _, damage_chip_mask = cv2.threshold(l_channel, 100, 255, cv2.THRESH_BINARY_INV)  # Shadows have low L values

    kernel2 = np.ones((5,5), np.uint8)  # 5x5 kernel for closing
    closed2 = cv2.morphologyEx(damage_chip_mask, cv2.MORPH_CLOSE, kernel2)

    input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    highlighted_image = input_image.copy()
    highlighted_image[damage_chip_mask == 255] = (255,0,0) 

    damaged_pixels = np.sum(damage_chip_mask==255)

    return damaged_pixels/total_pixels,highlighted_image

# Streamlit interface
st.markdown(
    """
    <style>
    .stApp h1 {
        text-align: center;
        background: linear-gradient(to right, #FDC41B, #F1CA51,#EFD773,#EBB04A,#750A04); /* Gradient palette */
        -webkit-background-clip: text;
        color: transparent;
        font-size: 5 rem; 
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Potato Chip Damage detection")

# Upload image
uploaded_image = st.file_uploader("Choose a potato chip image", type=['jpg', 'png', 'jpeg','webp'])

if uploaded_image is not None:
    # Create two columns for layout
    col1, col2 = st.columns(2)

    with col1:
        # Show the uploaded image in the first column
        st.image(uploaded_image, caption='Uploaded Image', width=300)

    try:
        # Preprocess the image and classify
        img_array = preprocess_image(uploaded_image)
        prediction = classify_image(img_array)
        damage_percentage, segmented_image = segment_image(uploaded_image)

        with col2:
            # Display the highlighted image in the second column
            st.image(segmented_image, caption='Highlighted Damage', width=300)

        # Display prediction result and damage percentage
        if prediction[0] > 0.5:
            st.markdown(f"<h2 style='color:#75F94D;'>This chip is not damaged.</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='color:#ED0025;'>This chip is damaged with {damage_percentage:.2%} damage.</h2>", unsafe_allow_html=True)

    except Exception as e:
        st.write(f"Error: {e}")
