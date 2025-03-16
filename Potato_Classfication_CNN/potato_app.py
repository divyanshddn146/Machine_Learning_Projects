import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


st.markdown(
    """
    <style>
    /* Background color or image */
    body {
        background-image: url('https://img.freepik.com/free-photo/potatoes-side-with-copy-space_23-2148540404.jpg?t=st=1737475874~exp=1737479474~hmac=fa02114b3eacda947d036ef5b49ada0300c331b9a222c3fb16af37f088fdbca4&w=996');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    
    /* Style for the main content area */
    .stApp {
        background-color: rgba(255, 255, 255, 0.1); /* Slightly transparent white */
    }
    
    /* Center alignment for the content */
    .stApp > div {
        display: flex;
        justify-content: center;
        align-items: center;

    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load your trained model
model = tf.keras.models.load_model('C:/Users/divya/Desktop/Machine_Learning_Sebestian/Potato_Classfication_CNN/Potato.keras')

# Function to preprocess the uploaded image
def preprocess_image(uploaded_image):
    img = Image.open(uploaded_image)
    img = img.resize((128, 128))  # Resize to match your model input
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

# Function to classify the image
def classify_image(img_array):
    prediction = model.predict(img_array)
    return prediction

# Function to resize the image while maintaining aspect ratio
def resize_image(uploaded_image, target_height):
    img = Image.open(uploaded_image)
    aspect_ratio = img.width / img.height
    target_width = int(target_height * aspect_ratio)
    resized_img = img.resize((target_width, target_height))  # Resize while keeping aspect ratio
    return resized_img

# Streamlit interface
st.markdown(
    """
    <style>
    .rainbow-text {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(to right,#e8690a,#fce4ac,#a47b49,#ccab93,#614632);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
    <h1 class="rainbow-text">Potato (Fresh v/s Rotten)</h1>
    """,
    unsafe_allow_html=True,
)

# Upload an image
uploaded_image = st.file_uploader("Choose a potato image", type=['jpg', 'png', 'jpeg'])

if uploaded_image is not None:
    # Resize the image for consistent display height
    resized_image = resize_image(uploaded_image, target_height=400)  # Fix the height to 300px

    # Preprocess the image and get the prediction
    img_array = preprocess_image(uploaded_image)
    prediction = classify_image(img_array)
    
    # Calculate confidence for both Fresh and Rotten
    fresh_confidence = prediction[0][0] * 100  # Fresh %
    rotten_confidence = 100 - fresh_confidence  # Rotten %

    # Determine the result
    is_fresh = prediction[0][0] > 0.5
    result_text = "This is a Fresh Potato" if is_fresh else "This is a Rotten Potato"
    result_color = "#32CD32" if is_fresh else "red"

    # Layout with columns
    col1, col2 = st.columns([1, 2])  # Adjust column widths as needed

    with col1:
        
        # Display the resized image with fixed height
        st.image(resized_image, caption="Uploaded Image", use_container_width=True)  
    
    with col2:
        if is_fresh:
            st.markdown(
                f"<h2 style='text-align: center;'>Fresh: {fresh_confidence:.2f}%</h2>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<h2 style='color:	#FCFFEB;text-align: center;'>Rotten: {rotten_confidence:.2f}%</h2>",
                unsafe_allow_html=True,
            )

    # Display the result in the center below the image and confidence
    st.markdown(
        f"<h2 style='text-align: center; color: {result_color};'>{result_text}</h2>",
        unsafe_allow_html=True,
    )

    