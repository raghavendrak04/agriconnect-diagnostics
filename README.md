# Agriconnect Diagnostics 🌱

### Advanced AI-Powered Crop Disease Diagnosis System
**Version 2.0**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red)
![Models](https://img.shields.io/badge/AI-TensorFlow%20%7C%20PyTorch-orange)
![LLM](https://img.shields.io/badge/LLM-Google%20Gemini-green)

Agriconnect is a state-of-the-art diagnostic tool designed to empower Pomegranate and Mango farmers. By leveraging a **Dual-Model** approach (MobileNet V2 + EfficientNet B4), it provides high-accuracy disease identification. Integrated with **Google Gemini**, it serves as a 24/7 intelligent farming assistant.

---

## 🚀 Key Features

### 🧠 Dual-Core Diagnostics
- **Speed & Precision**: Combines the speed of **MobileNet V2** (Keras) with the deep diagnostic power of **EfficientNet B4** (PyTorch).
- **Consensus System**: Automatically cross-verifies results from both models to minimize false positives.
- **Visual Confidence**: Displays confidence bars and inference latency for full transparency.

### 🤖 Intelligent Chat Assistant
- **Powered by Google Gemini**: A context-aware chatbot that understands agriculture.
- **Smart Suggestions**: Offers one-click questions like "How to treat Bacterial Blight?"
- **Actionable Advice**: Provides organic and chemical treatment plans instantly.

### 🎨 Modern Experience
- **Premium UI**: Glassmorphism design, smooth animations, and a clean, clutter-free interface.
- **Detailed Insights**: Rich cards displaying symptoms, characteristics, treatments, and management tips for every diagnosis.

---

## 🛠️ Technology Stack

- **Frontend**: Streamlit, Custom HTML/CSS
- **Deep Learning**: TensorFlow (Keras), PyTorch, Torchvision
- **GenAI**: Google Generative AI (Gemini Flash)
- **Image Processing**: Pillow (PIL), NumPy

---

## 📂 Project Structure

A modular architecture ensuring scalability and maintainability.

```
agriconnect-diagnostics/
├── components/          # UI Components (Hero, Cards, Details)
├── modules/             # Core Logic (Chatbot, Diagnostics, Data Loader)
├── utils/               # Utilities (Styles, File Helpers)
├── main.py              # Application Entry Point
├── info.json            # Disease Knowledge Base
├── requirements.txt     # Dependencies
└── Version_2_Report.md  # Detailed Technical Documentation
```

---

## ⚡ Getting Started

### Prerequisites
- Python 3.9 or higher installed.
- A Google Gemini API Key (for chatbot functionality).

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/raghavendrak04/agriconnect-diagnostics.git
   cd agriconnect-diagnostics
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Key**
   - Open `main.py`.
   - Update the `GEMINI_API_KEY` variable with your valid key.
   *(Note: For production, use environment variables)*

### Running the App

Double-click `run_app.bat` (Windows) or run:
```bash
streamlit run main.py
```

---

## 🎥 Usage Guide

1. **Upload**: Drag & Drop a leaf or fruit image into the analyzer zone.
2. **Analyze**: Click "Analyze Image" to trigger the dual-model inference.
3. **Review**: 
   - Check the **Consensus** status.
   - Read the detailed breakdown of symptoms and treatments.
4. **Consult**: Click the **Chatbot Icon** (bottom right) to ask follow-up questions about the diagnosis.

---

## 🔗 Resources

- **Live Demo**: [Streamlit Cloud Link](https://agriconnect-diagnostics-jgzinfgqd4x85heyxtbw4e.streamlit.app/)
- **Dataset**: [Google Drive Link](https://drive.google.com/drive/folders/1UjEWA42tInaJ_6PP1HSsAyooHVOBcbMM?usp=sharing)
- **Technical Report**: See [Version_2_Report.md](./Version_2_Report.md) for deep-dive technical details.

---

## 👨‍💻 Author

**Kurapati Raghavendra**  
- [LinkedIn Profile](https://www.linkedin.com/in/kurapati-raghavendra-39b3951b0/)
- [GitHub Repository](https://github.com/raghavendrak04/agriconnect-diagnostics)

---
*Empowering Farmers with AI.* 🌾
