import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import io
import os
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt

# Set up theme configuration
import pathlib

# Create .streamlit directory if it doesn't exist
config_dir = pathlib.Path(".streamlit")
config_dir.mkdir(exist_ok=True)

# Create config.toml with theme settings
config_path = config_dir / "config.toml"
config_text = """
[theme]
primaryColor = "#5D4037"  # Darker Brown (main brown color)
backgroundColor = "#FAF0E6"  # Linen (beige background)
secondaryBackgroundColor = "#DEB887"  # Burlywood (darker beige)
textColor = "#3E2723"  # Dark Brown for text
font = "sans serif"
"""
with open(config_path, "w") as f:
    f.write(config_text)

# Set page config
st.set_page_config(
    page_title="Kakao Bora",
    page_icon="🍫",
    layout="wide"
)

# App title and description with custom styling
st.markdown("<h1 style='color: #3E1C00;'>Kakao Bora</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #5D4037; font-size: 1.2em; font-style: italic;'>Your ultimate cocoa companion</p>", unsafe_allow_html=True)
st.write("Upload an image to detect cocoa contamination.")

# Class colors for visualization
class_colors = {
    0: (165, 42, 42),   # 'anthracnose' - brown
    1: (218, 165, 32),  # 'cssvd' - goldenrod
    2: (34, 139, 34)    # 'healthy' - forestgreen
}

# Class names
class_names = ['anthracnose', 'cssvd', 'healthy']

# Function to load the YOLO model
@st.cache_resource
def load_model():
    try:
        # Load model
        model = YOLO('cocoa_detection_model.pt')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Try to load the model
model = load_model()

if model:
    st.success("Model loaded successfully!")
else:
    st.error("Failed to load model. Please check the model path.")
    st.stop()

# Function to make predictions and visualize results
def predict_and_visualize(image):
    # Convert PIL Image to numpy array (if it's not already)
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
        
    # Run inference
    results = model(img_array)
    
    # Get the first result (assuming batch size of 1)
    result = results[0]
    
    # Create a copy of the original image for drawing
    img_with_boxes = Image.fromarray(img_array.copy())
    draw = ImageDraw.Draw(img_with_boxes)
    
    # Initialize class predictions
    class_preds = []
    
    # Draw bounding boxes and labels
    if hasattr(result, 'boxes') and len(result.boxes) > 0:
        for box, cls_idx, conf in zip(
            result.boxes.xyxy.cpu().numpy(), 
            result.boxes.cls.cpu().numpy(),
            result.boxes.conf.cpu().numpy()
        ):
            x1, y1, x2, y2 = box
            class_id = int(cls_idx)
            confidence = conf
            class_name = class_names[class_id]
            color = class_colors.get(class_id, (0, 0, 255))
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label
            label = f"{class_name} {confidence:.2f}"
            draw.rectangle([x1, y1 - 20, x1 + len(label) * 10, y1], fill=color)
            draw.text((x1, y1 - 20), label, fill=(255, 255, 255))
            
            # Add to class predictions
            class_preds.append((class_name, confidence))
    
    return img_with_boxes, class_preds

# Upload image functionality
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Make prediction on button click
    if st.button('Detect Contamination'):
        with st.spinner('Processing...'):
            # Get prediction and visualized image
            try:
                result_image, class_predictions = predict_and_visualize(image)
                
                with col2:
                    st.subheader("Prediction Results")
                    st.image(result_image, caption='Detection Results', use_container_width=True)
                
                # Display class predictions
                if class_predictions:
                    st.subheader("Detected Classes")
                    for class_name, confidence in class_predictions:
                        if class_name == 'healthy':
                            st.success(f"✅ {class_name.capitalize()}: {confidence:.2f}")
                        else:
                            st.error(f"⚠️ {class_name.capitalize()}: {confidence:.2f}")
                else:
                    st.info("No cocoa conditions detected in the image.")
            
            except Exception as e:
                st.error(f"Error during prediction: {e}")

# Add information about the model
with st.expander("About Kakao Bora"):
    st.write("""
    Kakao Bora uses a YOLOv8 deep learning model trained to detect contamination in cocoa.
    The model was trained on a dataset from the Zindi Amini Cocoa Contamination Challenge.
    
    The model can detect the following conditions:
    - 🔴 Anthracnose - A fungal disease affecting cocoa plants
    - 🟡 CSSVD (Cocoa Swollen Shoot Virus Disease) - A viral disease affecting cocoa
    - 🟢 Healthy - Cocoa plants without disease
    
    Upload an image of cocoa to get the contamination detection results.
    """)

# Add a sidebar with additional information
st.sidebar.title("Kakao Bora")
st.sidebar.image("https://i.pinimg.com/736x/46/b4/bf/46b4bf4dd1033ecdffe3b9fb7f4c08a3.jpg", use_container_width=True)
st.sidebar.markdown("""
## How to use
1. Upload an image of a cocoa plant or leaf
2. Click 'Detect Contamination'
3. View the results with bounding boxes

## About the model
This app uses a YOLOv8 object detection model trained to detect different cocoa conditions including diseases and healthy plants.
""")
