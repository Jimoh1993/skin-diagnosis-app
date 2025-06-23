# app.py

import streamlit as st
from PIL import Image
import io
import os
from datetime import datetime
import textwrap
from classifier import predict_with_gradcam
from llm_response import generate_initial_response, generate_followup_response
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# === PDF REPORT GENERATION ===
def generate_pdf_report(condition, confidence, age, skin_type, symptoms, llm_text, cam_path):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    margin = 40
    y = height - margin

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "🩺 AI-Powered Skin Diagnosis Report")
    y -= 30

    c.setFont("Helvetica", 9)
    c.drawString(margin, y, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 25

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, f"Predicted Condition: {condition}")
    y -= 18

    c.setFont("Helvetica", 11)
    c.drawString(margin, y, f"Confidence: {round(confidence * 100)}%")
    y -= 25

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Patient Information:")
    y -= 18

    c.setFont("Helvetica", 11)
    c.drawString(margin, y, f"- Age: {age}")
    y -= 15
    c.drawString(margin, y, f"- Skin Type: {skin_type}")
    y -= 15
    c.drawString(margin, y, f"- Symptoms: {symptoms if symptoms else 'N/A'}")
    y -= 25

    # Grad-CAM image
    try:
        if cam_path and os.path.exists(cam_path):
            c.drawImage(cam_path, margin, y - 200, width=width - 2 * margin, height=200, preserveAspectRatio=True)
            y -= 210
    except Exception as e:
        print(f"⚠️ Failed to add Grad-CAM image: {e}")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "AI Recommendations:")
    y -= 18

    c.setFont("Helvetica", 11)
    for line in textwrap.wrap(llm_text, width=95):
        if y < 40:
            c.showPage()
            y = height - margin
            c.setFont("Helvetica", 11)
        c.drawString(margin, y, line)
        y -= 14

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer


# === Streamlit UI ===
st.set_page_config(page_title="AI Skin Diagnosis", page_icon="🩺")
st.title("🩺 AI-Powered Skin Disease Diagnosis with Dermatology Recommendations")

# Session State Init
for key in [
    "condition", "confidence", "cam_path", "llm_response", "followup_response",
    "age", "skin_type", "symptoms"
]:
    if key not in st.session_state:
        st.session_state[key] = ""

uploaded_file = st.file_uploader("Upload a skin image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)
with col1:
    st.session_state.age = st.text_input("Patient Age", value=st.session_state.age)
    st.session_state.skin_type = st.text_input("Skin Type (e.g., oily, dry)", value=st.session_state.skin_type)
    st.session_state.symptoms = st.text_area("Symptoms or Notes", value=st.session_state.symptoms)

with col2:
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="📷 Uploaded Skin Image", use_container_width=True)
        except Exception as e:
            st.error(f"Invalid image file: {e}")
            st.stop()
    else:
        st.info("Please upload a skin image to begin diagnosis.")

# --- Diagnosis ---
if st.button("🧠 Diagnose Skin Condition") and uploaded_file:
    if not st.session_state.age.isdigit():
        st.error("⚠️ Please enter a valid numeric age.")
        st.stop()

    with st.spinner("Running AI diagnosis and generating explanation..."):
        try:
            condition, confidence, cam_path = predict_with_gradcam(image)
            llm_text = generate_initial_response(
                condition, confidence, st.session_state.age,
                st.session_state.skin_type, st.session_state.symptoms
            )

            st.session_state.condition = condition
            st.session_state.confidence = confidence
            st.session_state.cam_path = cam_path
            st.session_state.llm_response = llm_text

            st.success(f"✅ Diagnosis: **{condition}** ({confidence*100:.2f}% confidence)")
            st.image(cam_path, caption="Region of Interest (ROI) Heatmap", use_container_width=True)

            st.markdown("### 🧬 AI Explanation & Recommendation")
            st.write(llm_text)

        except Exception as e:
            st.error(f"❌ Diagnosis failed: {e}")

# --- Follow-up Question ---
if st.session_state.condition:
    st.markdown("---")
    st.header("💬 Ask a Follow-up Question")

    followup_q = st.text_area("Your follow-up question (e.g., treatment options, lifestyle advice):")

    if st.button("💡 Get Answer") and followup_q.strip():
        with st.spinner("Generating LLM response..."):
            try:
                response = generate_followup_response(
                    st.session_state.condition,
                    st.session_state.confidence,
                    st.session_state.age,
                    st.session_state.skin_type,
                    followup_q
                )
                st.session_state.followup_response = response
                st.markdown("### ✅ Follow-up Answer")
                st.write(response)
            except Exception as e:
                st.error(f"❌ Failed to generate response: {e}")

# --- PDF Download ---
if st.session_state.condition:
    st.markdown("---")
    st.header("📤 Download Diagnosis Report")

    pdf = generate_pdf_report(
        st.session_state.condition,
        st.session_state.confidence,
        st.session_state.age,
        st.session_state.skin_type,
        st.session_state.symptoms,
        st.session_state.llm_response,
        st.session_state.cam_path
    )

    st.download_button(
        label="📄 Download PDF Report",
        data=pdf,
        file_name=f"diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf"
    )

# --- Reset ---
if st.button("🔁 Reset All"):
    for key in st.session_state.keys():
        st.session_state[key] = ""
    st.experimental_rerun()
