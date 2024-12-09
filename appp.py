import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Streamlit page configuration
st.set_page_config(
    page_title="CIFAR-10 Classifier",
    page_icon="🚀",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for yellow background and black text
st.markdown("""
    <style>
    /* Set yellow background and black text globally */
    body {
        background-color: #FFD700; /* Yellow background */
        color: black; /* Black text */
    }
    .stApp {
        background-color: #FFD700; /* Yellow background */
    }
    h1, h2, h3, p, label, .stFileUploader label, .sidebar-content {
        color: black; /* Black text for all headings, labels, and paragraphs */
    }
    .stButton>button {
        background-color: black; /* Black button */
        color: yellow; /* Yellow text for buttons */
        font-size: 16px;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
    }
    .stButton>button:hover {
        background-color: yellow; /* Yellow button on hover */
        color: black; /* Black text on hover */
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🚀 CIFAR-10 Classifier")
st.write("Upload an image to see how the model classifies it among CIFAR-10 categories!")  # Text is black

# Load pre-trained model
@st.cache_resource
def load_cnn_model():
    return load_model("modell.h5")  # Replace with your model's file path

model = load_cnn_model()

# CIFAR-10 Class Labels
class_labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# File uploader
uploaded_file = st.file_uploader("📂 Upload an image (JPG or PNG)", type=["jpg", "png"])  # Text is black

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("⏳ Processing...")

    # Preprocess the image
    img = load_img(uploaded_file, target_size=(32, 32))  # CIFAR-10 image size
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    # Display the result
    st.markdown(f"### 🎯 Predicted Class: **{predicted_class}**")
    st.bar_chart(prediction[0], height=200, width=600)

# Sidebar with about info
st.sidebar.header("About the App")
st.sidebar.write("""
This application uses a Convolutional Neural Network (CNN) to classify images into CIFAR-10 categories:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

**💡 Tip:** Upload images resembling these categories to see the model in action.
""")
