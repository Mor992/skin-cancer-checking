import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import os
import gdown  # for downloading from Google Drive

st.title("Skin Cancer Detection using ResNet and Grad-CAM (2 Classes)")

# Path to your model file
model_path = "resnet_skin_cancer_2class.h5"

# Google Drive file ID from your link
GOOGLE_DRIVE_FILE_ID = "19hZJjhFR7kNUS7ZDbCMghnf3hwr4w37c"

@st.cache_resource
def load_my_model():
    if not os.path.exists(model_path):
        with st.spinner("Downloading model from Google Drive..."):
            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
            gdown.download(url, model_path, quiet=False)
    return tf.keras.models.load_model(model_path)

model = load_my_model()

# The class names (adjust if needed)
class_names = ['cancer', 'not_cancer']

def get_gradcam(model, img_array, layer_name):
    # Build a model that maps input -> activations of target layer + predictions
    target_layer = None
    try:
        # first try nested 'resnet50' layer
        target_layer = model.get_layer('resnet50').get_layer(layer_name)
    except Exception:
        target_layer = model.get_layer(layer_name)
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[target_layer.output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    cam = tf.maximum(cam, 0)
    max_val = tf.reduce_max(cam)
    if max_val != 0:
        cam = cam / max_val
    cam = tf.image.resize(cam[..., tf.newaxis], (img_array.shape[1], img_array.shape[2]))
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

    # Compute Grad-CAM
    cam = get_gradcam(model, img_array, "conv5_block3_out")

    fig, ax = plt.subplots()
    ax.imshow(loaded_img)
    ax.imshow(cam.squeeze(), cmap='jet', alpha=0.5)
    ax.axis('off')
    st.pyplot(fig)
