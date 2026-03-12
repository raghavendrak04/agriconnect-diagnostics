# Agriconnect V2 - Technical Report & Codebase Documentation

## 1. Executive Summary
Agriconnect V2 is an advanced agricultural diagnostic tool designed to help farmers identify diseases in Pomegranate and Mango crops. This version represents a significant architectural overhaul, moving from a monolithic script to a modular, scalable application. It leverages a dual-model Artificial Intelligence system (combining TensorFlow and PyTorch) to ensure high-accuracy diagnostics and integrates Generative AI (Google Gemini) for an interactive expert farming assistant.

## 2. Technology Stack

### Core Frameworks
- **Language**: Python 3.9+
- **Web Framework**: Streamlit (for rapid, interactive data app development)

### Artificial Intelligence & Machine Learning
- **Deep Learning Frameworks**: 
  - **TensorFlow/Keras**: Runs the MobileNet V2 model (optimized for speed).
  - **PyTorch**: Runs the EfficientNet B4 model (optimized for accuracy).
- **Generative AI**: 
  - **Google Gemini API**: Powers the "Agriconnect Assistant" chatbot for natural language advice.
- **Computer Vision**: 
  - **Pillow (PIL)**: Image processing and manipulation.
  - **NumPy**: Numerical operations and array handling.

### Frontend & UI
- **Styling**: Native Streamlit components augmented with Custom CSS (injected via `utils/styles.py`) for a premium, glassmorphism-inspired look.
- **Animations**: CSS animations for fade-ins and smooth transitions.

## 3. System Architecture

The application follows a **Modular Architecture**, separating logic into distinct layers:
1.  **Presentation Layer (`main.py`, `components/`)**: Handles UI rendering and user interaction.
2.  **Logic Layer (`modules/`)**: Contains core functionality like model inference, data loading, and chatbot capability.
3.  **Data Layer (`info.json`)**: Static JSON database containing disease descriptions, treatments, and symptoms.
4.  **Utility Layer (`utils/`)**: Helper functions for styling and file management.

## 4. Codebase Explanation

### Directory Structure
```
agriconnect-diagnostics/
├── main.py                 # Entry point of the application
├── info.json               # Knowledge base for crops and diseases
├── requirements.txt        # Project dependencies
├── components/             # UI Building Blocks
│   └── ui_components.py    # Hero section, cards, detailed views
├── modules/                # Core Business Logic
│   ├── chatbot.py          # Gemini AI integration
│   ├── data_loader.py      # JSON data parsing utility
│   └── diagnostics.py      # TF & PyTorch model inference logic
└── utils/                  # Helper Utilities
    ├── file_utils.py       # Binary file handling (e.g., images to base64)
    └── styles.py           # Custom CSS injection
```

### Module Breakdown

#### 1. `main.py` (The Orchestra)
This is the central controller. It:
- Configures the Streamlit page.
- Initializes session state for the chatbot.
- Loads the ML models at startup.
- Manages the flow: Application of CSS -> Header Render -> Image Upload -> Analysis Trigger -> Result Display.
- Implements the **Consensus Logic**: It compares predictions from both models. If they match, it verifies the result ("Consensus Reached"). If they differ, it presents both options to the user.

#### 2. `modules/diagnostics.py` (The Brain)
- **`load_keras_model` & `load_pytorch_model`**: Uses `@st.cache_resource` to load heavy model files into memory only once, speeding up subsequent reloads.
- **`predict_keras`**: Preprocesses images for MobileNet (224x224, float32) and runs inference.
- **`predict_pytorch`**: Preprocesses images for EfficientNet (Resize to 380x380, Normalization) and runs inference.
- **Returns**: Prediction class index, confidence score, and inference time (latency).

#### 3. `modules/chatbot.py` (The Assistant)
- **`configure_gemini`**: Sets up the Google GenAI client.
- **`get_chatbot_response`**: definite the system prompt. It instructs Gemini to act as an agricultural expert, limiting its scope to Pomegranate and Mango farming to prevent hallucinations or irrelevant answers.
- **`render_chatbot`**: Manages the chat UI, history (session state), and predefined suggestion chips (e.g., "How to treat Bacterial Blight?").

#### 4. `modules/data_loader.py` (The Knowledge)
- Loads `info.json` efficiently.
- **`get_info_by_class_name`**: A robust search function that finds disease details by matching prediction labels against various keys in the JSON (Disease Name, Stage Name, Titles), handling variability in naming conventions.

#### 5. `components/ui_components.py` (The Face)
- Contains modular rendering functions like `render_hero` (header), `render_overview` (feature grid), and `render_prediction_cards` (visual bars for model confidence).
- Keeps `main.py` clean by abstracting HTML/CSS embedding.

## 5. Key Features of Version 2

- **Dual-Model Verification**: By running two different architectures (MobileNet & EfficientNet) in parallel, the system provides a "Second Opinion" automatically, significantly reducing false positives.
- **Smart Consensus**: The UI clearly signals when models match (High Confidence) versus when they disagree (Manual Review needed).
- **Proactive AI Assistant**: The chatbot isn't just a passive tool; the UI prompts users to "Ask the Expert" specifically when they view a disease diagnosis, creating a seamless workflow from Problem Identified -> Solution Found.
- **Enterprise-Grade UI**: The move to custom CSS and modular components makes the app look like a modern SaaS product rather than a basic data script.

## 6. Future Roadmap
- **Database Integration**: Move from `info.json` to a proper SQL database for managing dynamic content.
- **User Authentication**: Allow farmers to save their scan history.
- **Multilingual Support**: Add localization for regional languages (e.g., Gujarati, Hindi).
