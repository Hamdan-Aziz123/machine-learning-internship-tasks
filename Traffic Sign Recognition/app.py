import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd

# Load the trained model
model = load_model("traffic_sign_model.h5")

# Manually defined class names
class_names = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons"
}

# Image Preprocessing
def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (32, 32))
    img = img / 255.0  # normalize
    img = np.expand_dims(img, axis=0)  # shape: (1, 32, 32, 3)
    return img

# Streamlit UI
st.title("🚦 Traffic Sign Recognition")

st.write("""
Upload an image of a traffic sign, and the model will predict its class.
""")

uploaded_file = st.file_uploader("Upload Traffic Sign Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=300)

    # Predict button
    if st.button("Predict"):
        img_array = preprocess_image(image)
        prediction = model.predict(img_array)
        class_id = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        # Show results
        st.success(f"Predicted Class ID: {class_id}")
        st.info(f"Class Name: {class_names.get(class_id, 'Unknown')}")
        st.write(f"Confidence: {confidence:.2f}")
