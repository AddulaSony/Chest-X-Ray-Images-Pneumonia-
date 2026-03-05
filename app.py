import streamlit as st
import numpy as np
from PIL import Image
from model import load_model, predict_image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Pneumonia Detection", layout="centered")

# -------------------------------
# 1️⃣ Header Section
# -------------------------------

st.title("🩺 Pneumonia Detection from Chest X-Ray")
st.subheader("AI Assisted Screening Tool")

st.write(
"""
Upload a chest X-ray image to check pneumonia risk.
This AI system analyzes the uploaded X-ray and predicts the probability
of pneumonia using a trained CNN model.
"""
)

st.markdown("---")

# -------------------------------
# 2️⃣ File Upload Section
# -------------------------------

uploaded_file = st.file_uploader(
    "Upload X-ray Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file)
    except:
        st.error("❌ Invalid image file. Please upload JPG or PNG.")
        st.stop()

    # -------------------------------
    # 3️⃣ Image Preview Section
    # -------------------------------

    st.subheader("🖼 Image Preview")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Original Image")
        st.image(image, use_column_width=True)

    resized_image = image.resize((150,150))

    with col2:
        st.write("Resized Image (Model Input)")
        st.image(resized_image, use_column_width=True)

    st.markdown("---")

    # -------------------------------
    # 4️⃣ Prediction Button
    # -------------------------------

    if st.button("🔍 Analyze Image"):

        with st.spinner("Analyzing X-ray..."):

            model = load_model()

            label, probability = predict_image(image, model)

            pneumonia_prob = probability
            normal_prob = 1 - probability

        st.markdown("---")

        # -------------------------------
        # A️⃣ Prediction Result Card
        # -------------------------------

        st.subheader("📊 Prediction Result")

        if label == "PNEUMONIA":
            st.error(f"Status: {label}")
        else:
            st.success(f"Status: {label}")

        st.write(f"Confidence: {probability*100:.2f}%")

        # -------------------------------
        # B️⃣ Probability Display
        # -------------------------------

        st.subheader("Probability Distribution")

        prob_data = {
            "Normal": normal_prob,
            "Pneumonia": pneumonia_prob
        }

        st.bar_chart(prob_data)

        st.write(f"Normal: {normal_prob*100:.2f}%")
        st.write(f"Pneumonia: {pneumonia_prob*100:.2f}%")

        # -------------------------------
        # C️⃣ Risk Indicator
        # -------------------------------

        st.subheader("⚠ Risk Indicator")

        if pneumonia_prob > 0.8:
            risk = "HIGH"
            st.error("⚠️ Risk Level: HIGH")

        elif pneumonia_prob > 0.5:
            risk = "MODERATE"
            st.warning("⚠️ Risk Level: MODERATE")

        else:
            risk = "LOW"
            st.success("✅ Risk Level: LOW")

        st.markdown("---")

        # -------------------------------
        # D️⃣ Disclaimer Section
        # -------------------------------

        st.subheader("⚠ Disclaimer")

        st.info(
        """
        This AI tool is for educational purposes only.

        The prediction should not be considered as a medical diagnosis.
        Please consult a certified radiologist or healthcare professional
        for accurate medical evaluation.
        """
        )

