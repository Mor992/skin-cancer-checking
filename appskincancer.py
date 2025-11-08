%%writefile app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import os
import gdown # Import gdown for downloading from Google Drive

st.title("Skin Cancer Detection using ResNet and Grad-CAM (2 Classes)")

# Path to your model file
model_path = "resnet_skin_cancer_2class.h5"

# Replace with your actual Google Drive File ID (ensure file is shared publicly)
GOOGLE_DRIVE_FILE_ID = "19hZJjhFR7kNUS7ZDbCMghnf3hwr4w37c" 

# Load the 2-class model
@st.cache_resource
def load_my_model():
    if not os.path.exists(model_path) and GOOGLE_DRIVE_FILE_ID != "19hZJjhFR7kNUS7ZDbCMghnf3hwr4w37c":
        with st.spinner("Downloading model from Google Drive..."):
            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
            gdown.download(url, model_path, quiet=False) # quiet=False to show progress
    elif not os.path.exists(model_path):
        st.error("Model not found locally. Please ensure 'resnet_skin_cancer_2class.h5' is in the same directory, or update GOOGLE_DRIVE_FILE_ID.")
        st.stop()
    return tf.keras.models.load_model(model_path)

model = load_my_model()

# Define class names for the 2-class model (assuming alphabetical order from generator)
class_names = ['cancer', 'not_cancer'] # Verify this order after training

def get_gradcam(model, img_array, layer_name):
    # Access the sub-layer within the 'resnet50' layer
    if model.get_layer('resnet50') is not None:
        target_layer = model.get_layer('resnet50').get_layer(layer_name)
    else:
        target_layer = model.get_layer(layer_name)

    grad_model = tf.keras.models.Model(
        inputs=model.input, outputs=[target_layer.output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, tf.argmax(predictions[0])]
    grads = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.zeros(conv_outputs.shape[1:3], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_outputs[0, :, :, i]
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)
    cam = tf.image.resize(cam[..., np.newaxis], (img_array.shape[1], img_array.shape[2]))
    return cam.numpy()

uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    loaded_img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(loaded_img)
    img_array = np.expand_dims(img_array / 255.0, axis=0)
    prediction = model.predict(img_array, verbose=0)
    pred_class_idx = np.argmax(prediction[0])
    pred_class_name = class_names[pred_class_idx]

    st.image(loaded_img, caption=f"Predicted Class: {pred_class_name}", use_column_width=True)

    # Assuming 'conv5_block3_out' is still the appropriate layer for ResNet50
    cam = get_gradcam(model, img_array, "conv5_block3_out")
    fig, ax = plt.subplots()
    ax.imshow(loaded_img)
    ax.imshow(cam.squeeze(), cmap='jet', alpha=0.5)
    ax.axis('off')
    st.pyplot(fig)
    
