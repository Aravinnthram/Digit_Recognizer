import numpy as np
import cv2
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def preprocess_image(image_path):
    """
    Preprocess the input image for the Random Forest model.
    This includes resizing, grayscaling, and flattening the image.
    """
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    
    # Resize the image to the required size (e.g., 8x8 for digit recognition)
    image_resized = cv2.resize(image, (8, 8))
    
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

def predict(image_path, model_path):
    """
    Perform inference using the Random Forest model on the input image.
    """
    # Preprocess the image
    image_features = preprocess_image(image_path)
    
    # Load the trained model
    model = load_model(model_path)
    
    # Perform prediction
    prediction = model.predict([image_features])
    
    return prediction[0]

if __name__ == "__main__":
    # Example usage
    image_path = "digit_image.jpg"  # Replace with the path to your input image
    model_path = "digit_model.pkl" # Replace with the path to your saved model
    
    try:
        result = predict(image_path, model_path)
        print(f"Predicted class: {result}")
    except Exception as e:
        print(f"Error during inference: {e}")