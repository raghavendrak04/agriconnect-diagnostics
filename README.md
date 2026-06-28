<div align="center">

# 🌱 Agriconnect — Crop Diagnostics Platform

### AI-Powered Disease Diagnosis for Pomegranate & Mango Farmers

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Gemini](https://img.shields.io/badge/Google_Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)

**Upload a crop photo → Get instant AI diagnosis → Receive expert treatment advice**

[Live Demo](https://agriconnect-diagnostics-jgzinfgqd4x85heyxtbw4e.streamlit.app/) · [Dataset](https://drive.google.com/drive/folders/1UjEWA42tInaJ_6PP1HSsAyooHVOBcbMM?usp=sharing) · [LinkedIn](https://www.linkedin.com/in/kurapati-raghavendra-39b3951b0/)

</div>

---

## 📖 What is Agriconnect?

Agriconnect is a **farmer-first AI diagnostic platform** that helps Pomegranate and Mango farmers identify crop diseases instantly by uploading a photo. It uses a **Dual-Model AI** approach — running **MobileNet V2** (TensorFlow) and **EfficientNet B4** (PyTorch) simultaneously — providing a built-in "second opinion" that builds farmer trust.

The platform also includes an **AI Chat Assistant** powered by **Google Gemini 2.0 Flash** that acts as a 24/7 agricultural expert, giving practical treatment recommendations in natural language.

### 🎯 12 Detectable Classes

| Category | Classes |
|:---|:---|
| 🦠 **Diseases** | Alternaria · Anthracnose · Bacterial Blight · Cercospora |
| 🌿 **Growth Stages** | Bud · Flower · Early-Fruit · Mid-Growth · Ripe |
| ✅ **Health Status** | Healthy |
| 🥭 **Mango Varieties** | Kesar · Calypso |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      USER's BROWSER                         │
│                                                             │
│  ┌───────────────┐         ┌─────────────────────────────┐  │
│  │  Landing Page  │  ────►  │    Diagnostics Tool          │  │
│  │  (Hero, CTA)  │         │  (Upload, Results, Chat)     │  │
│  └───────────────┘         └─────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           │  HTTP REST API
┌──────────────────────────▼──────────────────────────────────┐
│                FastAPI Backend  (app.py)                     │
│                                                             │
│  API Endpoints:                                             │
│  ├── GET  /                  → Landing page                 │
│  ├── GET  /diagnostics       → Diagnostics tool             │
│  ├── GET  /api/model-status  → Model loading status         │
│  ├── POST /api/load-models   → Background model loading     │
│  ├── POST /api/analyze       → Dual-model image inference   │
│  └── POST /api/chat          → Gemini AI conversation       │
│                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ TensorFlow  │  │   PyTorch    │  │  Gemini 2.0 Flash │  │
│  │ MobileNetV2 │  │ EfficientB4  │  │  (Direct HTTP)    │  │
│  │ (224×224)   │  │ (380×380)    │  │                   │  │
│  └─────────────┘  └──────────────┘  └───────────────────┘  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │      info.json  —  Knowledge Base (12 entries)      │    │
│  │  Diseases · Growth Stages · Mango Varieties         │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Dual-Model Inference Flow

```
     User uploads image
             │
             ▼
    ┌─────────────────┐
    │ POST /api/analyze│
    └────────┬────────┘
             │
      ┌──────┴──────┐
      ▼              ▼
 ┌─────────┐   ┌──────────┐
 │  Keras   │   │ PyTorch  │
 │ MobileNet│   │ EffNetB4 │
 │ (224×224)│   │ (380×380)│
 └────┬─────┘   └─────┬────┘
      │                │
      └──────┬─────────┘
             ▼
      ┌─────────────┐
      │  Consensus?  │
      │ Both agree?  │
      └──────┬───────┘
          ┌──┴──┐
          ▼     ▼
       ✅ Yes  ⚠️ No
       Unified  Both shown
       result   with tabs
```

---

## 🧠 Model Training Configuration

### Model 1: MobileNet V2 (TensorFlow/Keras)

> **Weight File:** `best_85_plus_model_tf.h5` (~27 MB)  
> **Training Notebook:** `pom-mango-classification-mobilenet.ipynb`

| Parameter | Value |
|:---|:---|
| **Base Architecture** | `tf.keras.applications.MobileNetV2` |
| **Pre-trained Weights** | ImageNet (Transfer Learning) |
| **Input Size** | 224 × 224 × 3 |
| **Output Classes** | 12 (softmax) |
| **Optimizer** | Adam |
| **Loss Function** | `categorical_crossentropy` |
| **Batch Size** | 64 (head training) → 32 (fine-tuning) |
| **Learning Rate** | 0.001 (initial) → 0.0001 (fine-tune) → 1e-5 (deep fine-tune) |
| **LR Scheduler** | `ReduceLROnPlateau` (factor=0.3, patience=3) |
| **Early Stopping** | patience=5, monitor=`val_accuracy` |
| **Checkpoint** | `ModelCheckpoint` saving best `val_accuracy` |
| **Preprocessing** | `mobilenet_v2.preprocess_input` (scales to [-1, 1]) |
| **Class Weights** | ✅ Used (handles imbalanced dataset) |
| **Training Strategy** | 3-phase: Frozen base (10 epochs) → Unfreeze top layers (15 epochs) → Full fine-tune (50 epochs) |
| **Best Val Accuracy** | ~85%+ |

### Model 2: EfficientNet B4 (PyTorch)

> **Weight File:** `Pom-mango_EfficientNet_pytorch.pth` (~71 MB)  
> **Training Notebook:** `Pom-G&D_EfficientNet.ipynb`

| Parameter | Value |
|:---|:---|
| **Base Architecture** | `torchvision.models.efficientnet_b4` |
| **Pre-trained Weights** | None (trained from scratch) |
| **Input Size** | 224 × 224 (training) → 380 × 380 (production inference) |
| **Output Classes** | 12 (`model.classifier[1] = Linear(1792, 12)`) |
| **Optimizer** | Adam (lr=0.001) |
| **Loss Function** | `nn.CrossEntropyLoss` |
| **Batch Size** | 16 |
| **LR Scheduler** | `ReduceLROnPlateau` (factor=0.3, patience=3) |
| **Early Stopping** | Manual implementation (patience=5) |
| **Epochs** | 50 max (with early stopping) |
| **Normalization** | ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |
| **Data Augmentation** | RandomHorizontalFlip, RandomRotation, ColorJitter |
| **Inference Mode** | `torch.no_grad()` + `model.eval()` |

### Other Trained Models (Experimental — Not in Production)

| Notebook | Model | Key Config | Status |
|:---|:---|:---|:---|
| `Pom-MobileNet.ipynb` | MobileNet V2 (PyTorch) | Adam lr=0.001, batch=64, epochs=50, `ReduceLROnPlateau` | Experiment |
| `Pom-G&D_ConvNext-large.ipynb` | ConvNeXt-Large (PyTorch) | **AdamW** lr=1e-4, **weight_decay=1e-4**, `CosineAnnealingLR` T_max=20 | Experiment |
| `Pom-ConvNext-large.ipynb` | ConvNeXt-Large (PyTorch) | Same as above | Experiment |
| `Pom-EfficientNet.ipynb` | EfficientNet B4 (PyTorch) | Adam lr=0.001, batch=64, epochs=50 | Experiment |
| `pom-mango-classification-efficientnet.ipynb` | EfficientNetV2-B0 (TF) | IMAGE_SIZE=256×256, batch=32, 50 epochs, categorical_crossentropy | **61.84% test acc** |
| `convert-to-tflite.ipynb` | TFLite Conversion | Dynamic quantization & float32 exports | Utility |

### Training Data Augmentation (Keras Pipeline)

```python
ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=mobilenet_v2.preprocess_input
)
```

---

## 📂 Project Structure & File Guide

```
source/
│
├── app.py                          # ⭐ FastAPI backend (V3 — production)
├── main.py                         # ⭐ Streamlit entry point (V2 — modular)
├── info.json                       # ⭐ Disease/Stage/Variety knowledge base
├── requirements.txt                # ⭐ Python dependencies
├── run_app.bat                     # ⭐ Windows quick-start script
├── chatbot_icon.png                # ⭐ Floating chatbot button icon
│
├── Agriconnect_MVP_V2_V3_Iteration_Plan.md   # 📄 Full iteration roadmap V1→V3
├── Agriconnect_V3_MVP_Report.md              # 📄 V3 product report + user study
├── Version_2_Report.md                       # 📄 V2 technical documentation
│
├── pom-mango-classification-mobilenet.ipynb  # 🧪 MobileNetV2 (TF) training
├── pom-mango-classification-efficientnet.ipynb # 🧪 EfficientNetV2-B0 (TF) training
├── Pom-G&D_EfficientNet.ipynb                # 🧪 EfficientNet B4 (PyTorch) training
├── Pom-MobileNet.ipynb                       # 🧪 MobileNet V2 (PyTorch) training
├── Pom-G&D_ConvNext-large.ipynb              # 🧪 ConvNeXt-Large experiment
├── Pom-ConvNext-large.ipynb                  # 🧪 ConvNeXt-Large experiment
├── Pom-EfficientNet.ipynb                    # 🧪 EfficientNet B4 (PyTorch) alt
├── convert-to-tflite.ipynb                   # 🧪 TFLite conversion utility
│
├── best_85_plus_model_tf.h5                  # 🏋️ MobileNetV2 weights (~27 MB)
├── Pom-mango_EfficientNet_pytorch.pth        # 🏋️ EfficientNet B4 weights (~71 MB)
├── best_efficientnet_final.h5                # 🏋️ EfficientNetV2-B0 TF (~71 MB)
├── best_efficientnet_tf.h5                   # 🏋️ EfficientNetV2-B0 TF (~71 MB)
├── best_model_tf.h5                          # 🏋️ MobileNetV2 alt weights (~28 MB)
├── pom_mango_*_quant.tflite                  # 🏋️ TFLite quantized models
├── pom_mango_*_float32.tflite                # 🏋️ TFLite float32 models
│
└── (scratch notebooks: pom.ipynb, pom2.ipynb, etc.)  # 🗑️ Training experiments
```

---

## 📑 Detailed File Descriptions

### ⭐ `app.py` — FastAPI Backend (V3 Production Server)

> **The production backend** — a complete REST API server with dual-model inference and Gemini-powered chatbot.

| Aspect | Detail |
|:---|:---|
| **Framework** | FastAPI + Uvicorn |
| **CORS** | Open (`*`) for development |
| **Model Loading** | Lazy — background thread with `threading.Lock` for thread safety |
| **Inference** | Dual-model: Keras MobileNet V2 (224×224) + PyTorch EfficientNet B4 (380×380) |
| **Chat** | Direct HTTP POST to Gemini 2.0 Flash API (no SDK — avoids protobuf conflicts) |

**Key Design Decisions:**
- Models load in a **background thread** to avoid blocking startup — the frontend polls `/api/model-status` until ready
- Gemini integration uses **direct HTTP calls** instead of the `google-generativeai` SDK to avoid protobuf version conflicts with TensorFlow
- `get_info()` uses **fuzzy matching** (exact → substring → reverse substring) to handle naming inconsistencies between model class labels and `info.json` keys

**API Endpoints:**

| Method | Path | Description |
|:---|:---|:---|
| `GET` | `/` | Serves landing page |
| `GET` | `/diagnostics` | Serves diagnostics tool |
| `GET` | `/api/model-status` | Returns: `not_loaded` / `loading` / `loaded` / `error` |
| `POST` | `/api/load-models` | Triggers background model loading thread |
| `POST` | `/api/analyze` | Image upload → dual-model inference → consensus check |
| `POST` | `/api/chat` | Text message → Gemini with agricultural system prompt |

---

### ⭐ `main.py` — Streamlit Entry Point (V2 Modular App)

> **The V2 Streamlit application** — an earlier version with modular imports from `modules/`, `components/`, `utils/`. Still functional and deployable via `streamlit run main.py`.

| Aspect | Detail |
|:---|:---|
| **Framework** | Streamlit |
| **Model Caching** | `@st.cache_resource` for one-time model loading |
| **Chat** | `google-generativeai` SDK via `modules/chatbot.py` |
| **UI** | Custom glassmorphism CSS, floating chatbot popover |

---

### ⭐ `info.json` — Disease & Crop Knowledge Base

> A JSON array of **12 entries** covering diseases, growth stages, and mango varieties.

Each entry contains:

| Field | Example |
|:---|:---|
| `Category` | `"Disease"` / `"Growth Stage"` / `"Variety"` |
| `Disease_Name` / `Stage_Name` | `"Bacterial Blight"` / `"Bud"` |
| `Description` | Scientific description of condition |
| `Symptoms` | `"Water-soaked lesions, dark angular spots..."` |
| `Treatment` | `"Spray copper oxychloride at 3g/L..."` |
| `MANAGEMENT_TIPS` | `"Remove infected plant parts..."` |
| `Prevention` | `"Use disease-free planting material"` |
| `DISEASE_Severity_Value` | `"4"` (0–4 scale) |
| `Images` | Array of reference image URLs |

---

### ⭐ `requirements.txt` — Dependencies

```
fastapi
uvicorn
tensorflow
torch
torchvision
Pillow
```

### ⭐ `run_app.bat` — Windows Quick Start

Launches the FastAPI server with a single double-click.

### ⭐ `chatbot_icon.png` — Chatbot Button Icon

The floating action button icon for the AI chat interface.

---

### 📄 Documentation Files

| File | Lines | Content |
|:---|:---:|:---|
| `Agriconnect_MVP_V2_V3_Iteration_Plan.md` | 768 | Full V1→V2→V3 roadmap, MoSCoW prioritization, 10-user feedback study, SWOT analysis |
| `Agriconnect_V3_MVP_Report.md` | 682 | V3 product report, system architecture, 6 screenshots, user personas |
| `Version_2_Report.md` | 93 | V2 modular architecture, module breakdown, feature descriptions |

---

### 🧪 Training Notebooks

| Notebook | Model | Framework | Purpose |
|:---|:---|:---|:---|
| `pom-mango-classification-mobilenet.ipynb` | MobileNet V2 | TensorFlow | **Production model** — 3-phase transfer learning, 85%+ accuracy |
| `pom-mango-classification-efficientnet.ipynb` | EfficientNetV2-B0 | TensorFlow | EfficientNet TF experiment — 61.84% test accuracy |
| `Pom-G&D_EfficientNet.ipynb` | EfficientNet B4 | PyTorch | **Production model** — 12-class classification |
| `Pom-MobileNet.ipynb` | MobileNet V2 | PyTorch | PyTorch MobileNet experiment |
| `Pom-G&D_ConvNext-large.ipynb` | ConvNeXt-Large | PyTorch | Large model experiment (AdamW + CosineAnnealing) |
| `Pom-ConvNext-large.ipynb` | ConvNeXt-Large | PyTorch | ConvNeXt variant experiment |
| `Pom-EfficientNet.ipynb` | EfficientNet B4 | PyTorch | EfficientNet alt experiment |
| `convert-to-tflite.ipynb` | — | TensorFlow | TFLite conversion (dynamic quantization + float32) |

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
|:---|:---|:---|
| **Backend** | Python 3.10, FastAPI, Uvicorn | REST API server |
| **ML Model 1** | TensorFlow/Keras — MobileNet V2 | Speed-optimized inference (224×224) |
| **ML Model 2** | PyTorch — EfficientNet B4 | Accuracy-optimized inference (380×380) |
| **Chat AI** | Google Gemini 2.0 Flash (HTTP) | Natural language farming advice |
| **Frontend** | HTML5, CSS3, Vanilla JS | Full UI — no framework lock-in |
| **Image Processing** | Pillow (PIL), NumPy | Preprocessing for inference |
| **Concurrency** | `threading.Lock` + background threads | Thread-safe model loading |

---

## ⚡ Getting Started

### Prerequisites

- Python 3.9+
- ~200MB disk space for model weights

### Installation

```bash
# Clone the source branch
git clone -b source https://github.com/raghavendrak04/agriconnect-diagnostics.git
cd agriconnect-diagnostics

# Install dependencies
pip install -r requirements.txt
```

### Download Model Files

> ⚠️ Model weight files (`.h5`, `.pth`) are **not included** in this repository due to size limits. Download them:
>
> 📦 [Google Drive — Model Files](https://drive.google.com/drive/folders/1UjEWA42tInaJ_6PP1HSsAyooHVOBcbMM?usp=sharing)

Place downloaded files in the project root:
- `best_85_plus_model_tf.h5` — Keras MobileNet V2 weights
- `Pom-mango_EfficientNet_pytorch.pth` — PyTorch EfficientNet B4 weights

### Run

**Option A: FastAPI (V3 — Production)**
```bash
python app.py
# Open http://localhost:8080
```

**Option B: Streamlit (V2)**
```bash
streamlit run main.py
# Opens automatically in browser
```

**Option C: Windows**
```bash
# Double-click run_app.bat
```

---

## 📊 Version Evolution

| Capability | V1 | V2 | V3 |
|:---|:---:|:---:|:---:|
| Single-model AI diagnosis | ✅ | ✅ | ✅ |
| Dual-model consensus engine | ❌ | ✅ | ✅ |
| AI chat assistant (Gemini) | ❌ | ✅ | ✅ |
| Modular codebase | ❌ | ✅ | ✅ |
| Premium glassmorphism UI | ❌ | ✅ | ✅ |
| Professional landing page | ❌ | ❌ | ✅ |
| RESTful JSON API | ❌ | ❌ | ✅ |
| Lazy background model loading | ❌ | ❌ | ✅ |
| Mobile-first responsive design | ❌ | ❌ | ✅ |

---

## 🚫 Files Not in Repo (& Why)

| File | Size | Reason |
|:---|:---|:---|
| `*.h5` | 27–71 MB each | Model weights exceed GitHub limits |
| `*.pth` | ~71 MB | PyTorch weights exceed GitHub limits |
| `*.tflite` | 3–23 MB each | TFLite converted models |
| `*.ipynb` | — | Training notebooks (experiments, not production code) |
| `main_backup.py` | 20 KB | Monolithic V1 backup, superseded by modular V2 |
| `check_models.py` | <1 KB | Debug script |
| `test_gemini_candidates.py` | <1 KB | Test script |

> 📦 Download all model weights from [Google Drive](https://drive.google.com/drive/folders/1UjEWA42tInaJ_6PP1HSsAyooHVOBcbMM?usp=sharing)

---

## 👨‍💻 Author

**Kurapati Raghavendra**  
📧 veeraraghavendra.k22@iiits.in  
🔗 [LinkedIn](https://www.linkedin.com/in/kurapati-raghavendra-39b3951b0/) · [GitHub](https://github.com/raghavendrak04)

---

<div align="center">

*Empowering Farmers with AI* 🌾

</div>
