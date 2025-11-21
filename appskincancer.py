"""
Streamlit app: Grad-CAM for a Keras ResNet model
Files included: this single-file app. Place final_resnet_model.keras in /mnt/data/ or provide the Google Drive file ID.

Usage (development):
    streamlit run streamlit_gradcam_app.py

Features:
- Load model from local path or Google Drive (via gdown)
- Upload image or use sample
- Compute Grad-CAM heatmap and overlay on image
- Show top-k predictions
- Option to provide class names (comma-separated)

Defaults provided for convenience:
- default local path: /mnt/data/final_resnet_model.keras
- default Google Drive ID: 1kJWpQQlF-2Rtwj2xRmVtbDw-83cyjD3q

Note: this app expects a Keras model saved with `model.save("...")` compatible with tf.keras.models.load_model.
"""

import streamlit as st
import numpy as np
from PIL import Image
import io
import os
import tensorflow as tf
import matplotlib.pyplot as plt

# Optional: to download from Google Drive
try:
    import gdown
    GDOWN_AVAILABLE = True
except Exception:
    GDOWN_AVAILABLE = False

# ---------------------
# Utility functions
# ---------------------

def load_model_from_local(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    model = tf.keras.models.load_model(path)
    return model


def download_model_from_gdrive(file_id, dest_path):
    if not GDOWN_AVAILABLE:
        raise RuntimeError("gdown is not installed. Add gdown to requirements or install it.")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=False)
    return dest_path


def preprocess_input_image(img: Image.Image, target_size=(224,224)):
    img = img.convert('RGB')
    img = img.resize(target_size)
    arr = np.array(img).astype(np.float32)
    # Use ResNet preprocessing if available
    try:
        from tensorflow.keras.applications.resnet50 import preprocess_input
        arr = preprocess_input(arr)
    except Exception:
        arr = (arr / 255.0)
    return arr


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # img_array: (H,W,3) preprocessed
    img_tensor = tf.expand_dims(img_array, axis=0)

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()
    return heatmap


def overlay_heatmap_on_image(original_img: Image.Image, heatmap, alpha=0.4, cmap='jet'):
    import matplotlib.cm as cm

    heatmap_resized = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(original_img.size, resample=Image.BILINEAR)
    heatmap_arr = np.array(heatmap_resized)

    colormap = cm.get_cmap(cmap)
    colored_heatmap = colormap(heatmap_arr / 255.0)[:, :, :3]
    colored_heatmap = (colored_heatmap * 255).astype(np.uint8)
    colored_heatmap = Image.fromarray(colored_heatmap)

    overlay = Image.blend(original_img.convert('RGBA'), colored_heatmap.convert('RGBA'), alpha=alpha)
    return overlay

# ---------------------
# Streamlit UI
# ---------------------

st.set_page_config(page_title='Grad-CAM for ResNet', layout='centered')
st.title('Grad-CAM viewer (ResNet-compatible)')

st.markdown("Provide a Keras `.keras` model saved with `model.save(...)` (or let the app download the model from Google Drive).")

# defaults per user input
DEFAULT_LOCAL_MODEL_PATH = "/mnt/data/final_resnet_model.keras"
DEFAULT_GDRIVE_ID = "1kJWpQQlF-2Rtwj2xRmVtbDw-83cyjD3q"

use_download = st.checkbox('Download model from Google Drive (if unchecked, load local file)', value=False)
model = None
model_load_error = None

if use_download:
    file_id = st.text_input('Google Drive file ID', value=DEFAULT_GDRIVE_ID)
    dest = st.text_input('Destination path to save model', value=os.path.join('/tmp','final_resnet_model.keras'))
    if st.button('Download and load model'):
        try:
            st.info('Downloading model...')
            download_model_from_gdrive(file_id, dest)
            st.success('Download complete; loading model...')
            model = load_model_from_local(dest)
            st.success('Model loaded successfully')
        except Exception as e:
            model_load_error = str(e)
            st.error(f'Error loading model: {model_load_error}')
else:
    local_path = st.text_input('Local model path', value=DEFAULT_LOCAL_MODEL_PATH)
    if st.button('Load local model'):
        try:
            model = load_model_from_local(local_path)
            st.success('Model loaded successfully')
        except Exception as e:
            model_load_error = str(e)
            st.error(f'Error loading model: {model_load_error}')

# Show model summary (collapsed)
if model is not None:
    with st.expander('Show model summary'):
        buf = io.StringIO()
        try:
            model.summary(print_fn=lambda x: buf.write(x + "\n"))
            st.text(buf.getvalue())
        except Exception as e:
            st.text('Unable to show model summary: ' + str(e))

# Image input
st.header('Input image')
uploaded = st.file_uploader('Upload an image', type=['jpg','jpeg','png'])
use_sample = st.button('Use sample image (if no upload)')

if uploaded is not None:
    image = Image.open(uploaded)
elif use_sample:
    # create a blank placeholder or instruct user
    st.info('No sample included. Please upload an image.')
    image = None
else:
    image = None

# Class names input
class_names_input = st.text_input('Comma-separated class names (optional). Example: benign,malignant', value='')
class_names = [c.strip() for c in class_names_input.split(',') if c.strip()]

# Last convolutional layer name input (try common ResNet names)
last_conv_default = st.text_input('Last convolutional layer name (ResNet example: conv5_block3_out)', value='conv5_block3_out')

# Run Grad-CAM
if st.button('Run Grad-CAM'):
    if model is None:
        st.error('Load a model first')
    elif image is None:
        st.error('Upload an image first')
    else:
        try:
            # Preprocess
            target_size = (model.input_shape[1], model.input_shape[2]) if len(model.input_shape) >= 3 else (224,224)
            pre = preprocess_input_image(image, target_size=target_size)

            # Predict
            preds = model.predict(np.expand_dims(pre, axis=0))
            top_k = min(5, preds.shape[-1])
            top_inds = preds[0].argsort()[-top_k:][::-1]

            # Display predictions
            st.subheader('Top predictions')
            for i, idx in enumerate(top_inds):
                name = class_names[idx] if idx < len(class_names) else str(idx)
                st.write(f"{i+1}. {name} — score: {float(preds[0, idx]):.4f}")

            # Grad-CAM
            last_conv_layer_name = last_conv_default
            try:
                heatmap = make_gradcam_heatmap(pre, model, last_conv_layer_name, pred_index=top_inds[0])
            except Exception as e:
                st.error('Error computing Grad-CAM: ' + str(e))
                st.stop()

            overlay = overlay_heatmap_on_image(image, heatmap, alpha=0.4)

            # Show images
            st.image(image, caption='Original image', use_column_width=True)
            st.image(overlay, caption='Grad-CAM overlay', use_column_width=True)

            # Also show raw heatmap
            fig, ax = plt.subplots()
            ax.axis('off')
            ax.imshow(heatmap, interpolation='nearest')
            st.pyplot(fig)

        except Exception as e:
            st.error('Failed to run Grad-CAM: ' + str(e))

# Footer / tips
st.markdown('''
**Tips:**
- If you get `Layer not found` for the last conv layer, open model summary and find a suitable final convolution block name (e.g. `conv5_block3_out` for ResNet50 as provided by Keras applications).
- If predictions look reversed or inputs are wrong, check preprocessing: the app tries `tf.keras.applications.resnet50.preprocess_input` first, otherwise scales to [0,1].
- To include this app on GitHub: place this file in a repo with `requirements.txt` listing dependencies (tensorflow, streamlit, pillow, numpy, matplotlib, gdown).
''')
