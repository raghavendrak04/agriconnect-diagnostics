import streamlit as st
import tensorflow as tf
import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image, ImageOps
import json
import os
import time

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Agriconnect",
    page_icon="🍃",
    layout="wide"
)

# --- 2. CONSTANTS ---
CLASS_NAMES = ['Alternaria', 'Anthracnose', 'Bacterial_Blight', 'Calypso', 'Cercospora', 'Healthy', 'Kesar', 'bud', 'early-fruit', 'flower', 'mid-growth', 'ripe']
KERAS_MODEL_PATH = "best_85_plus_model_tf.h5"
PYTORCH_MODEL_PATH = "Pom-mango_EfficientNet_pytorch.pth"

# --- 3. CUSTOM CSS ---
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
        color: #1a1a1a;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .prediction-card {
        background-color: white;
        color: #333333;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .prediction-value {
        color: #2e7d32;
        font-weight: bold;
        font-size: 1.1em;
    }
    .info-box {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 15px;
        border-radius: 4px;
        margin-top: 10px;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# --- 4. DATA LOADING HELPER ---
@st.cache_data
def load_info_data():
    try:
        with open('info.json', 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Error loading info.json: {e}")
        return []

info_data = load_info_data()

def get_info_by_class_name(class_name):
    """
    Finds the info.json entry matching the predicted class name.
    Matches against Stage_Title, Disease_Title, Variety_Title using substring or direct mapping.
    """
    if not class_name: return None
    
    # Normalize class name for matching
    search_term = class_name.lower().replace("_", " ") 
    
    for item in info_data:
        # Check various fields
        candidates = [
            item.get('Stage_Name', ''),
            item.get('Disease_Name', ''),
            item.get('Variety_Name', ''),
            item.get('Stage_Title', ''),
            item.get('Disease_Title', ''),
            item.get('Variety_Title', '')
        ]
        
        # 1. Exact match (case insensitive)
        if any(c.lower() == search_term for c in candidates):
            return item
            
        # 2. Check if class_name is IN the candidate (e.g. 'Calypso' in 'Mango___Calypso')
        # OR if candidate is IN the class_name
        if any(search_term in c.lower() for c in candidates) or any(c.lower() in search_term for c in candidates):
            return item
            
    return None

# --- 5. MODEL LOADING HELPERS ---

@st.cache_resource
def load_keras_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading Keras model {model_path}: {e}")
        return None

@st.cache_resource
def load_pytorch_model(path):
    try:
        # Architecture: EfficientNet-B4
        model = models.efficientnet_b4(weights=None)
        
        # Adjust classifier for 12 classes
        model.classifier[1] = torch.nn.Linear(1792, 12)
        
        # Load weights
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading PyTorch model: {e}")
        return None

# --- 6. PREDICTION LOIGC ---

def predict_keras(model, image):
    # Resize to expected size (usually 224x224 for standard models)
    target_size = (224, 224) 
    image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    
    if len(img_array.shape) == 3:
        img_array = np.expand_dims(img_array, axis=0) # (1, 224, 224, 3)
        
    # Scale if needed (Usually 1./255 for standard Keras models)
    img_array = img_array.astype(np.float32) / 255.0
    
    start_time = time.time()
    predictions = model.predict(img_array, verbose=0)
    end_time = time.time()
    
    idx = np.argmax(predictions)
    conf = np.max(predictions)
    
    return idx, conf, (end_time - start_time) * 1000

def predict_pytorch(model, image):
    # EfficientNet-B4 resolution
    transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = transform(image).unsqueeze(0)
    
    start_time = time.time()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    end_time = time.time()
    
    confidence, prediction_index = torch.max(probabilities, 0)
    
    return prediction_index.item(), confidence.item(), (end_time - start_time) * 1000

# --- 7. UI HELPER FUNCTIONS ---

def render_details(data):
    if not data: 
        st.warning("No details found for this class in database.")
        return
    
    category = data.get('Category', 'Unknown')
    title = data.get('Stage_Name') or data.get('Disease_Name') or data.get('Variety_Name')
    st.markdown(f"## {title} <span style='font-size:0.6em; color:gray'>({category})</span>", unsafe_allow_html=True)
    
    with st.expander("📖 Description & Key Info", expanded=True):
        st.write(data.get('Description', 'No description available.'))
        
    c1, c2 = st.columns(2)
    with c1:
        if 'Symptoms' in data:
            st.markdown("#### ⚠️ Symptoms")
            st.info(data['Symptoms'])
        if 'Characteristics' in data:
                st.markdown("#### ✨ Characteristics")
                st.info(data['Characteristics'])
        if 'Usage' in data:
                st.markdown("#### 🍽️ Usage")
                st.write(data['Usage'])
        if 'Affected_Plant_Parts' in data:
                st.markdown("#### 🌿 Affected Parts")
                st.write(data['Affected_Plant_Parts'])
                
    with c2:
        if 'Treatment' in data:
            st.markdown("#### 💊 Treatment")
            st.success(data['Treatment'])
        if 'Taste_Profile' in data:
                st.markdown("#### 😋 Taste Profile")
                st.write(data['Taste_Profile'])
        if 'Growing_Conditions' in data:
                st.markdown("#### ☀️ Growing Conditions")
                st.write(data['Growing_Conditions'])

    if 'MANAGEMENT_TIPS' in data:
        st.warning(f"**💡 Management Tips:** {data['MANAGEMENT_TIPS']}")
        
    if 'Prevention' in data:
        st.markdown(f"**🛡️ Prevention:** {data['Prevention']}")
        
    if 'Images' in data and isinstance(data['Images'], list):
        with st.expander("🖼️ Reference Images (from database)", expanded=False):
            cols = st.columns(min(len(data['Images']), 4))
            for idx, img_url in enumerate(data['Images'][:4]): # Show max 4
                with cols[idx]:
                    st.image(img_url, use_container_width=True)

# --- 8. MAIN APP LOGIC ---

def main():
    st.markdown("<h1 style='text-align: center;'>Agriconnect 🌱</h1>", unsafe_allow_html=True)
    st.title(" Pomegranate & Mango Diagnostics")
    st.write("Upload an image to identify the growth stage, disease, or variety.")

    # Sidebar for Model Info
    st.sidebar.header("Model Configuration")
    st.sidebar.info("Comparing Keras (.h5) and PyTorch (.pth) models.")

    # Load Models
    model_keras = load_keras_model(KERAS_MODEL_PATH)
    model_pytorch = load_pytorch_model(PYTORCH_MODEL_PATH)

    if model_keras and model_pytorch:
        st.sidebar.success("Models Loaded Successfully!")

    # File Uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display Image Centered
        image = Image.open(uploaded_file)
        
        # Create a centered column layout for the image
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(image, caption='Uploaded Image', use_container_width=True)
            
            # Analyze Button below image
            analyze_btn = st.button("🔍 Analyze Image", type="primary", use_container_width=True)

        if analyze_btn:
            st.divider()
            st.write("### 🔍 Analysis Results")
            
            if model_keras and model_pytorch:
                # Show a progress spinner
                with st.spinner('Running AI analysis...'):
                    progerss = st.progress(0)
                    
                    # 1. Keras Prediction
                    idx_keras, conf_keras, time_keras = predict_keras(model_keras, image)
                    name_keras = CLASS_NAMES[idx_keras] if idx_keras < len(CLASS_NAMES) else "Unknown"
                    res_keras = get_info_by_class_name(name_keras)
                    progerss.progress(50)

                    # 2. PyTorch Prediction
                    if image.mode != 'RGB':
                        image_rgb = image.convert('RGB')
                    else:
                        image_rgb = image
                    idx_pytorch, conf_pytorch, time_pytorch = predict_pytorch(model_pytorch, image_rgb)
                    name_pytorch = CLASS_NAMES[idx_pytorch] if idx_pytorch < len(CLASS_NAMES) else "Unknown"
                    res_pytorch = get_info_by_class_name(name_pytorch)
                    progerss.progress(100)
                    time.sleep(0.5) # Small delay for visual effect
                    progerss.empty()

                # Display Results Comparison
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.markdown(f"""
                    <div class="prediction-card" style="text-align: center; border-top: 5px solid #2196F3;">
                        <h3 style="color: #2196F3; margin-bottom: 0;">Best Model (Keras)</h3>
                        <p style="font-size: 0.9em; color: #666;">High Accuracy</p>
                        <hr>
                        <h2 class="prediction-value" style="font-size: 1.8em; margin: 10px 0;">{name_keras}</h2>
                        <p><strong>Confidence:</strong> {conf_keras:.1%}</p>
                        <p style="font-size: 0.8em; color: #999;">Inference: {time_keras:.1f} ms</p>
                    </div>
                    """, unsafe_allow_html=True)

                with res_col2:
                        st.markdown(f"""
                    <div class="prediction-card" style="text-align: center; border-top: 5px solid #E91E63;">
                        <h3 style="color: #E91E63; margin-bottom: 0;">EfficientNet (PyTorch)</h3>
                        <p style="font-size: 0.9em; color: #666;">Robust Verification</p>
                        <hr>
                        <h2 class="prediction-value" style="font-size: 1.8em; margin: 10px 0;">{name_pytorch}</h2>
                        <p><strong>Confidence:</strong> {conf_pytorch:.1%}</p>
                        <p style="font-size: 0.8em; color: #999;">Inference: {time_pytorch:.1f} ms</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Show Detailed Info (Prioritizing PyTorch or Agreement)
                st.divider()
                
                # Display Logic
                if name_keras == name_pytorch:
                    st.success(f"✅ **Consensus Reached:** Both models identify this as **{name_pytorch}**")
                    render_details(res_pytorch)
                else:
                    st.warning("⚠️ **Model Disagreement:** The models have different predictions.")
                    tab1, tab2 = st.tabs([f"Keras: {name_keras} (Best)", f"PyTorch: {name_pytorch}"])
                    
                    with tab1:
                        render_details(res_keras)
                        
                    with tab2:
                            render_details(res_pytorch)

            else:
                st.error("Models not loaded. Please check that all model files are present.")

    else:
        # Placeholder for when no image is uploaded
        st.info("👆 Please upload an image above to start the diagnostics.")

if __name__ == "__main__":
    main()
