import streamlit as st
import tensorflow as tf
import keras
import numpy as np
from matplotlib.image import imread
import colorsys
import cv2
import os
from scipy import ndimage as nd



st.markdown(
    """
    <style>
    /* Set the background image */
    .stApp {
        background-image: url('https://img.freepik.com/free-photo/flat-lay-beer-bottles-with-chips-nuts_23-2148754981.jpg?t=st=1737990572~exp=1737994172~hmac=e8b153437b6ef164d138d6caac73be5d1776fa91bd4d9e005e50eb6115f3bc3b&w=1060');
        background-size: cover;
        background-position: center;
    
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Load your trained model (use the appropriate format, .h5 or .keras)
model_path = os.path.join(os.path.dirname(__file__), 'model.keras')
model = tf.keras.models.load_model(model_path)

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
    hsv = cv2.cvtColor(input_image,cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv,(0,0,0),(180, 255, 65))

    closed_mask = nd.binary_closing(mask,np.ones((12,12)))

    mask_colored = np.zeros_like(input_image)
    mask_colored[closed_mask == 1] = [255, 0, 0]
    
    damaged_pixels = np.sum(closed_mask == 1)

    hsv_chip = cv2.cvtColor(input_image,cv2.COLOR_RGB2HSV)
    mask_chip = cv2.inRange(hsv,(20,50,110),(30, 255, 255))

    closed_mask_chip = nd.binary_closing(mask_chip,np.ones((3,3)))

    mask_colored_chip = np.zeros_like(input_image)
    mask_colored_chip[closed_mask_chip == 1] = [0, 255, 0]  

    overlay = cv2.addWeighted(input_image, 0.7, mask_colored, 0.3, 0)
    overlay = cv2.addWeighted(overlay, 1, mask_colored_chip, 0.3, 0)

    undamaged_pixels = np.sum(closed_mask_chip == 1)

    total_pixels = undamaged_pixels + damaged_pixels

    return damaged_pixels/total_pixels,overlay

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
        margin-top: -50px; 
        padding-top: 0px;
        padding-bottom: 50px; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Potato Chip Damage Detection")
test_images_1 = {
    "Sample Defective Chip 1": os.path.join(os.path.dirname(__file__), 'Pepsico/Test/Defective/IMG_20210319_004846.jpg'),
    "Sample Defective Chip 2": os.path.join(os.path.dirname(__file__), 'Pepsico/Test/Defective/IMG_20210319_010344.jpg'),
    "Sample Defective Chip 3": os.path.join(os.path.dirname(__file__), 'Pepsico/Test/Defective/IMG_20210319_004929.jpg'),
    "Sample Not Defective Chip 1": os.path.join(os.path.dirname(__file__), 'Pepsico/Test/Not Defective/IMG_20210318_231229.jpg'),
    "Sample Not Defective Chip 2": os.path.join(os.path.dirname(__file__), 'Pepsico/Test/Not Defective/IMG_20210318_231650.jpg'),
    "Sample Not Defective Chip 3": os.path.join(os.path.dirname(__file__), 'Pepsico/Test/Not Defective/IMG_20210318_232125.jpg')
}
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 300px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
# Sidebar for file upload
with st.sidebar:
    st.write("### Upload your potato chip image")
    uploaded_image = st.file_uploader("",type=['jpg', 'png', 'jpeg', 'webp'])

    st.write("### Or Select a Sample Image:")

    selected_test_image = None
    cols = 1
    for i, (label, path) in enumerate(test_images_1.items()):
        if st.button(f"🔍 {label}"):
            selected_test_image = path

    

# Determine the image to process
if uploaded_image:
    image_to_process = uploaded_image
elif selected_test_image:
    image_to_process = selected_test_image
else:
    image_to_process = None

# Process the selected/uploaded image
if image_to_process:
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image_to_process, caption="Selected Image", width=300)

    try:
        img_array = preprocess_image(image_to_process)
        prediction = classify_image(img_array)
        damage_percentage, segmented_image = segment_image(image_to_process)

        with col2:
            st.image(segmented_image, caption="Highlighted Damage", width=300)

        # Display classification result
        if prediction[0] > 0.5:
            st.markdown("<h2 style='color:#75F94D;'>This chip is not damaged.</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='color:#ED0025;'>This chip is damaged with {damage_percentage:.2%} damage.</h2>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")