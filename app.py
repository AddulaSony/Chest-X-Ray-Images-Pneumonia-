import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# import model from training file
from chest_x_ray_images_(pneumonia) import model

st.set_page_config(page_title="Pneumonia Detection", layout="centered")

st.title("🩺 Pneumonia Detection from Chest X-Ray")
st.subheader("AI Assisted Screening Tool")

st.write("Upload a chest X-ray image to check pneumonia risk.")

uploaded_file = st.file_uploader(
    "Upload X-ray Image",
    type=["jpg","png","jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")
    st.image(image, use_column_width=True)

    # preprocess image
    img = image.resize((224,224))
    img = np.array(img)

    if img.shape[-1] == 4:
        img = img[:,:,:3]

    img = img/255.0
    img = np.expand_dims(img, axis=0)

    if st.button("🔍 Analyze Image"):

        prediction = model.predict(img)[0][0]

        pneumonia_prob = float(prediction)
        normal_prob = 1 - pneumonia_prob

        if pneumonia_prob > 0.5:
            label = "PNEUMONIA"
            st.error("⚠️ Pneumonia Detected")
        else:
            label = "NORMAL"
            st.success("✅ Normal Chest X-ray")

        st.write("### Prediction Result")

        st.write(f"**Status:** {label}")
        st.write(f"**Pneumonia Probability:** {pneumonia_prob*100:.2f}%")
        st.write(f"**Normal Probability:** {normal_prob*100:.2f}%")

        st.write("### Risk Level")

        if pneumonia_prob > 0.8:
            st.error("⚠️ HIGH RISK")
        elif pneumonia_prob > 0.5:
            st.warning("⚠️ MODERATE RISK")
        else:
            st.success("LOW RISK")

)
