import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from streamlit_lottie import st_lottie

# --------------------------------------------
# 1. Load Lottie Animations
# --------------------------------------------
def load_lottieurl(url: str):
    """
    Loads a Lottie animation from a given URL.
    Returns the JSON for that animation if successful.
    """
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Lottie animation URLs (feel free to change)
LOTTIE_LOADING_URL = "https://assets2.lottiefiles.com/packages/lf20_93ww8cqh.json"  # Loading spinner
LOTTIE_SUCCESS_URL = "https://assets10.lottiefiles.com/packages/lf20_u4yrau.json"   # Success animation

# Convert URLs to JSON
lottie_loading = load_lottieurl(LOTTIE_LOADING_URL)
lottie_success = load_lottieurl(LOTTIE_SUCCESS_URL)

# --------------------------------------------
# 2. Load Your Trained Model
# --------------------------------------------
MODEL_PATH = "emotion_model.keras"  # Update if your file path is different
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# --------------------------------------------
# 3. Define Class Names
# --------------------------------------------
class_names = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

# --------------------------------------------
# 4. Prediction Function
# --------------------------------------------
def predict_emotion(captured_image: Image.Image):
    """
    Predicts emotion from a PIL Image.
    Forces the image to RGB and resizes to (224, 224).
    """
    # Convert to 3-channel RGB
    img = captured_image.convert("RGB")
    img = img.resize((224, 224))

    # Convert to array and normalize
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
    img_array = img_array / 255.0

    # Model inference
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    return predicted_class, confidence

# --------------------------------------------
# 5. Streamlit App
# --------------------------------------------
def main():
    st.set_page_config(page_title="Live Emotion Detection", layout="centered")
    st.title("Emotion Detection from Live Camera")
    st.write("Use your camera to capture an image and detect your emotion.")

    # (Optional) Show a Lottie animation at the top
    if lottie_loading:
        st_lottie(lottie_loading, height=100, width=100, key="loading_top")

    # Camera input widget (replaces file uploader)
    snapshot = st.camera_input("Take a picture")

    if snapshot is not None:
        # Display the captured snapshot
        st.image(snapshot, caption="Captured Image", use_container_width=True)

        # Button to trigger prediction
        if st.button("Predict Emotion"):
            with st.spinner("Analyzing..."):
                # Show loading animation while analyzing
                if lottie_loading:
                    st_lottie(lottie_loading, height=150, width=150, key="analyzing_cam")

                # Convert snapshot (UploadedFile) to PIL Image
                captured_image = Image.open(snapshot)
                emotion, confidence = predict_emotion(captured_image)

            # Show results
            st.success(f"Predicted Emotion: {emotion}")
            st.info(f"Confidence: {confidence:.2f}")
            st.balloons()  # Built-in Streamlit balloons

            # Optional success animation
            if lottie_success:
                st_lottie(lottie_success, height=200, width=200, key="success_cam")

if __name__ == "__main__":
    main()
