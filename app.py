import streamlit as st
import numpy as np
import cv2
import pickle

# Suppress scikit-learn warnings about pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def preprocess_image(image):
    """
    Preprocess the input image for the Random Forest model.
    This includes resizing, grayscaling, and flattening the image.
    """
    # Convert the image to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize the image to the required size (e.g., 8x8 for digit recognition)
    image_resized = cv2.resize(image_gray, (8, 8))
    
    # Flatten the image to a 1D array
    image_flattened = image_resized.flatten()
    
    # Normalize the pixel values (optional, depending on your model training)
    image_normalized = image_flattened / 16.0  # Normalize to match training (e.g., 0-16 range for sklearn digits)
    
    return image_normalized

def load_model(model_path):
    """
    Load the trained Random Forest model from the specified path using pickle.
    """
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        raise ValueError(f"Failed to load model from {model_path}: {e}")

def predict(image, model):
    """
    Perform inference using the Random Forest model on the input image.
    """
    # Preprocess the image
    image_features = preprocess_image(image)
    
    # Perform prediction
    prediction = model.predict([image_features])
    
    return prediction[0]

# Streamlit app
st.title("Random Forest Image Classifier")
st.write("Upload an image to classify it using the trained Random Forest model.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")
    
    # Load the model
    model_path = "digit_model.pkl"  # Replace with the path to your saved model
    try:
        model = load_model(model_path)
        
        # Predict the class
        prediction = predict(image, model)
        st.write(f"Predicted Class: {prediction}")
    except Exception as e:
        st.error(f"Error during inference: {e}")