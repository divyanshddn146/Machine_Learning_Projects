import streamlit as st
import tensorflow as tf
import keras
import numpy as np

# Load your trained model (use the appropriate format, .h5 or .keras)
model = tf.keras.models.load_model('C:/Users/divya/Desktop/Machine_Learning_Sebestian/model.keras')  # or 'model_name.h5'

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

# Streamlit interface
st.title("Potato Chip Damage Detection")

# Upload image
uploaded_image = st.file_uploader("Choose a potato chip image", type=['jpg', 'png', 'jpeg'])

if uploaded_image is not None:
    # Show the uploaded image
    st.image(uploaded_image, caption='Uploaded Image', width=400)  # Set width to 500px for better display

    try:
        # Preprocess the image and classify
        img_array = preprocess_image(uploaded_image)
        prediction = classify_image(img_array)

        # Display prediction result with improved font size
        if prediction[0] > 0.5:
            st.markdown("<h2 style='color:green;'>This chip is not damaged.</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='color:red;'>This chip is damaged!</h2>", unsafe_allow_html=True)
    except Exception as e:
        st.write(f"Error: {e}")