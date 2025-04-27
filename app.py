import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Function to load the TensorFlow Lite model
def load_model(model_path):
    if not os.path.exists(model_path):
        raise ValueError(f"Model file '{model_path}' not found!")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Function to load labels from labels.txt
def load_labels(labels_path):
    with open(labels_path, 'r') as file:
        labels = file.read().splitlines()
    return labels

# Function to preprocess the image and predict
def preprocess_and_predict(interpreter, img, input_details, output_details):
    # Preprocess image
    img_array = image.img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array.astype(np.float32)  # Convert to float32
    
    # Normalize image (if your model expects it)
    img_array /= 255.0  # Normalize the image to the range [0, 1]

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run inference
    interpreter.invoke()

    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Streamlit UI setup
st.title("Anomaly Detection System 🌟")

# Check if the model file exists
model_path = 'model_unquant.tflite'  # Use the correct relative path here
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found! Please make sure the file is in the correct path.")
else:
    # Load the model and labels
    interpreter = load_model(model_path)  # Path to your TFLite model
    labels = load_labels('labels.txt')  # Path to your labels.txt

    # Get input and output details for TensorFlow Lite model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Image upload section
    st.write("Upload an image of plate Normal/Anomaly 📸")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        img = image.load_img(uploaded_image, target_size=(224, 224))  # Adjust size based on your model
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        # Prediction
        output_data = preprocess_and_predict(interpreter, img, input_details, output_details)
        
        # Get the prediction class and confidence
        predicted_class = np.argmax(output_data)  # Get the class with the highest score
        confidence = np.max(output_data)  # Get the confidence of the prediction
        
        # Display result
        if labels[predicted_class] == "Normal":
            st.write(f"Prediction: {labels[predicted_class]} with confidence {confidence:.2f} ")
        else:
            st.write(f"Prediction: {labels[predicted_class]} with confidence {confidence:.2f} ")

    # Real-time camera feed section
    st.write("Or use your camera to detect anomalies in real-time 📸")

    camera = st.camera_input("Capture Image for Anomaly Detection")

    if camera:
        st.image(camera, caption="Captured Image", use_column_width=True)
        
        # Convert captured image to the format expected by the model
        img = image.load_img(camera, target_size=(224, 224))  # Adjust size as needed
        output_data = preprocess_and_predict(interpreter, img, input_details, output_details)
        
        # Get the prediction class and confidence
        predicted_class = np.argmax(output_data)  # Get the class with the highest score
        confidence = np.max(output_data)  # Get the confidence of the prediction
        
        # Display prediction result
        if labels[predicted_class] == "Normal":
            st.write(f"Prediction: {labels[predicted_class]} with confidence {confidence:.2f} ")
        else:
            st.write(f"Prediction: {labels[predicted_class]} with confidence {confidence:.2f}")

    # Optional: Show instructions or any other information
    st.write("🌟 This is a simple Anomaly Detection System using TensorFlow Lite and Streamlit! 🌟")
