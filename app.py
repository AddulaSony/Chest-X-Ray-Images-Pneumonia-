%%writefile pneumonia_app.py

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("model.h5")

st.title("🩺 Pneumonia Detection from Chest X-Ray")
st.subheader("AI Assisted Screening Tool")

st.write("Upload a chest X-ray image to check pneumonia risk.")

# File upload
uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg","png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    resized = image.resize((150,150))
    img_array = np.array(resized)/255.0
    img_array = img_array.reshape(1,150,150,1)

    if st.button("🔍 Analyze Image"):

        prediction = model.predict(img_array)[0][0]

        pneumonia_prob = float(prediction)
        normal_prob = 1 - pneumonia_prob

        if pneumonia_prob > 0.5:
            label = "PNEUMONIA"
            color = "red"
        else:
            label = "NORMAL"
            color = "green"

        st.markdown("### Prediction Result")

        st.markdown(
            f"<h3 style='color:{color};'>Status: {label}</h3>",
            unsafe_allow_html=True
        )

        st.write(f"Confidence: {max(pneumonia_prob,normal_prob)*100:.2f}%")

        st.subheader("Probability")

        st.write(f"Normal: {normal_prob*100:.2f}%")
        st.write(f"Pneumonia: {pneumonia_prob*100:.2f}%")

        if pneumonia_prob > 0.8:
            risk = "HIGH"
        elif pneumonia_prob > 0.5:
            risk = "MODERATE"
        else:
            risk = "LOW"

        st.warning(f"⚠️ Risk Level: {risk}")

