import streamlit as st
from PIL import Image
import os
import time

# Import modular components
from modules.diagnostics import load_keras_model, load_pytorch_model, predict_keras, predict_pytorch
from modules.chatbot import configure_gemini, render_chatbot
from modules.data_loader import load_info_data, get_info_by_class_name
from components.ui_components import render_hero, render_overview, render_details, render_prediction_cards
from utils.styles import apply_custom_css
from utils.file_utils import get_base64_of_bin_file

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Agriconnect",
    page_icon="🍃",
    layout="wide"
)

# GEMINI CONFIGURATION
GEMINI_API_KEY = "AIzaSyBgndLh6fx_GIsfyFX4Zj335EFDR_iHKaw"
configure_gemini(GEMINI_API_KEY)

# --- 2. CONSTANTS ---
CLASS_NAMES = ['Alternaria', 'Anthracnose', 'Bacterial_Blight', 'Calypso', 'Cercospora', 'Healthy', 'Kesar', 'bud', 'early-fruit', 'flower', 'mid-growth', 'ripe']
KERAS_MODEL_PATH = "best_85_plus_model_tf.h5"
PYTORCH_MODEL_PATH = "Pom-mango_EfficientNet_pytorch.pth"

# Load essential data
info_data = load_info_data()

def main():
    # Initialize Session States
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Apply Custom CSS
    icon_base64 = get_base64_of_bin_file("chatbot_icon.png")
    apply_custom_css(icon_base64)

    # Render Header Section
    render_hero()
    render_overview()

    # Sidebar: Model Config & Quick Start
    st.sidebar.header("Model Configuration")
    st.sidebar.info("Comparing MobileNet V2 (Speed) and EfficientNet B4 (Accuracy).")
    
    # Load Models
    model_keras = load_keras_model(KERAS_MODEL_PATH)
    model_pytorch = load_pytorch_model(PYTORCH_MODEL_PATH)

    if model_keras and model_pytorch:
        st.sidebar.success("Models Ready ✅")

    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Quick Start")
    st.sidebar.markdown("""
    1. **Upload** a clear photo of a Pomegranate or Mango leaf/fruit.
    2. **Analyze** using our dual-model system.
    3. **Compare** results and get detailed treatment info.
    4. **Chat** with our AI to ask follow-up questions.
    """)

    st.sidebar.markdown("---")
    st.sidebar.header("📬 Connect")
    st.sidebar.markdown(
        """
        <div style="display: flex; flex-direction: column; gap: 12px;">
            <a href="https://www.linkedin.com/in/kurapati-raghavendra-39b3951b0/" target="_blank" style="text-decoration: none; color: inherit;">
                <div style="background: rgba(10, 102, 194, 0.1); padding: 12px; border-radius: 12px; border: 1px solid rgba(10, 102, 194, 0.2); text-align: center; font-weight: 600; color: #0a66c2;">
                    Connect on LinkedIn
                </div>
            </a>
            <a href="https://github.com/raghavendrak04/agriconnect-diagnostics" target="_blank" style="text-decoration: none; color: inherit;">
                <div style="background: rgba(0, 0, 0, 0.05); padding: 12px; border-radius: 12px; border: 1px solid rgba(0, 0, 0, 0.1); text-align: center; font-weight: 600;">
                    View Source Code
                </div>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

    # File Uploader Container
    st.markdown("<div id='analyzer'></div>", unsafe_allow_html=True)
    st.markdown("### 📸 Image Analyzer")
    uploaded_file = st.file_uploader("Drop your crop image here or click to browse", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(image, caption='Uploaded Image', use_container_width=True)
            analyze_btn = st.button("🔍 Analyze Image", type="primary", use_container_width=True)

        if analyze_btn:
            st.divider()
            st.write("### 🔍 Analysis Results")
            
            if model_keras and model_pytorch:
                with st.spinner('AI analyzing...'):
                    # Keras Prediction
                    idx_keras, conf_keras, time_keras = predict_keras(model_keras, image)
                    name_keras = CLASS_NAMES[idx_keras] if idx_keras < len(CLASS_NAMES) else "Unknown"
                    res_keras = get_info_by_class_name(info_data, name_keras)

                    # PyTorch Prediction
                    idx_pytorch, conf_pytorch, time_pytorch = predict_pytorch(model_pytorch, image)
                    name_pytorch = CLASS_NAMES[idx_pytorch] if idx_pytorch < len(CLASS_NAMES) else "Unknown"
                    res_pytorch = get_info_by_class_name(info_data, name_pytorch)

                # Display Results Comparison
                render_prediction_cards(name_keras, conf_keras, time_keras, name_pytorch, conf_pytorch, time_pytorch)

                st.divider()
                
                # Consensus Logic
                if name_keras == name_pytorch:
                    st.success(f"✅ **Consensus Reached:** Both models identify this as **{name_pytorch}**")
                    render_details(res_pytorch)
                else:
                    st.warning("⚠️ **Model Disagreement:** Models have different predictions.")
                    tab1, tab2 = st.tabs([f"Lighter: {name_keras} ", f"Heavier: {name_pytorch}"])
                    with tab1: render_details(res_keras)
                    with tab2: render_details(res_pytorch)
                
                # Proactive Chatbot Hint
                st.markdown("""
                <div style="background: #eff6ff; border: 1px solid #3b82f6; padding: 15px; border-radius: 12px; display: flex; align-items: center; gap: 15px; margin-top: 20px;">
                    <div style="font-size: 1.5rem;">🤖</div>
                    <div>
                        <div style="font-weight: 700; color: #1e3a8a;">Need a detailed action plan?</div>
                        <div style="font-size: 0.9rem; color: #1e40af;">Click the blue chatbot icon in the bottom right to talk to our expert AI!</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("Model error. Check files.")

    else:
        st.info("👆 Please upload an image above to start diagnostics.")

    # Render Floating Chatbot
    with st.popover(" ", help="Agriconnect Assistant"):
        render_chatbot()

if __name__ == "__main__":
    main()
