# Pomegranate & Mango Model Diagnostics
This repository contains a simple Streamlit application for classifying pomegranate and mango images using advanced Deep Learning models.

- live Server Link - https://agriconnect-diagnostics-jgzinfgqd4x85heyxtbw4e.streamlit.app/
- Dataset Link  - https://drive.google.com/drive/folders/1UjEWA42tInaJ_6PP1HSsAyooHVOBcbMM?usp=sharing

## Setup
1. **Install Python**: Ensure you have Python 3.9+ installed.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This will install Streamlit, TensorFlow, PyTorch, and other necessary libraries.*

## Running the App
Double-click `run_app.bat` or run the following command in your terminal:
```bash
streamlit run main.py
```

## Features
- **Dual Model Architecture**: 
  - **MobileNet (Lighter) Model **: High-accuracy primary classifier.
  - **EfficientNet (Heavier) Model **: EfficientNet-based robust verification model.
- **Consensus Logic**: Checks if both models agree on the diagnosis for higher confidence.
- **Detailed Insights**: Pulls rich metadata including symptoms, treatments, and management tips.
- **Modern UI**: Clean, vertical layout with progress indicators and clear result cards.

## Models Included
- `best_85_plus_model_tf.h5` (Keras)
- `Pom-mango_EfficientNet_pytorch.pth` (PyTorch)




