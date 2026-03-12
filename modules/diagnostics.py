import streamlit as st
import tensorflow as tf
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
import time

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
        model = models.efficientnet_b4(weights=None)
        model.classifier[1] = torch.nn.Linear(1792, 12)
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading PyTorch model: {e}")
        return None

def predict_keras(model, image):
    target_size = (224, 224) 
    image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    
    if len(img_array.shape) == 3:
        img_array = np.expand_dims(img_array, axis=0)
        
    img_array = img_array.astype(np.float32) / 255.0
    
    start_time = time.time()
    predictions = model.predict(img_array, verbose=0)
    end_time = time.time()
    
    idx = np.argmax(predictions)
    conf = np.max(predictions)
    
    return idx, conf, (end_time - start_time) * 1000

def predict_pytorch(model, image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
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
