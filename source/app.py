from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
import time
import io
import base64
import threading

app = FastAPI(title="Agriconnect Diagnostics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Constants ───────────────────────────────────────────────────────────────
CLASS_NAMES = [
    'Alternaria', 'Anthracnose', 'Bacterial_Blight', 'Calypso',
    'Cercospora', 'Healthy', 'Kesar', 'bud', 'early-fruit',
    'flower', 'mid-growth', 'ripe'
]
GEMINI_API_KEY = "AIzaSyBgndLh6fx_GIsfyFX4Zj335EFDR_iHKaw"

# ─── Lazy model state ────────────────────────────────────────────────────────
models = {"keras": None, "pytorch": None, "status": "not_loaded"}
model_lock = threading.Lock()

def load_models_background():
    global models
    with model_lock:
        if models["status"] == "loaded":
            return
        models["status"] = "loading"
        try:
            import tensorflow as tf
            keras_model = tf.keras.models.load_model("best_85_plus_model_tf.h5")
            models["keras"] = keras_model

            import torch
            import torchvision.models as tv_models
            pt_model = tv_models.efficientnet_b4(weights=None)
            pt_model.classifier[1] = torch.nn.Linear(1792, 12)
            state = torch.load("Pom-mango_EfficientNet_pytorch.pth", map_location="cpu")
            pt_model.load_state_dict(state)
            pt_model.eval()
            models["pytorch"] = pt_model

            models["status"] = "loaded"
        except Exception as e:
            models["status"] = f"error: {str(e)}"

# ─── Info data ───────────────────────────────────────────────────────────────
with open("info.json", "r") as f:
    INFO_DATA = json.load(f)

def get_info(class_name):
    if not class_name:
        return None
    term = class_name.lower().replace("_", " ")
    for item in INFO_DATA:
        candidates = [
            item.get('Stage_Name', ''), item.get('Disease_Name', ''),
            item.get('Variety_Name', ''), item.get('Stage_Title', ''),
            item.get('Disease_Title', ''), item.get('Variety_Title', '')
        ]
        valid = [c for c in candidates if c and c.strip()]
        if any(c.lower() == term for c in valid):
            return item
        if any(term in c.lower() for c in valid) or any(c.lower() in term for c in valid):
            return item
    return None

# ─── Routes ──────────────────────────────────────────────────────────────────
@app.get("/")
def index():
    return FileResponse("static/index.html")

@app.get("/diagnostics")
def diagnostics():
    return FileResponse("static/diagnostics.html")

@app.get("/api/model-status")
def model_status():
    return {"status": models["status"]}

@app.post("/api/load-models")
def trigger_load():
    if models["status"] == "not_loaded":
        thread = threading.Thread(target=load_models_background, daemon=True)
        thread.start()
    return {"message": "loading started", "status": models["status"]}

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if models["status"] != "loaded":
        # Trigger load if not started
        if models["status"] == "not_loaded":
            thread = threading.Thread(target=load_models_background, daemon=True)
            thread.start()
        raise HTTPException(status_code=503, detail={"status": models["status"], "message": "Models are still loading. Please wait..."})

    try:
        import numpy as np
        from PIL import Image, ImageOps
        import torch
        from torchvision import transforms

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # ── Keras prediction ──────────────────────────────────────────
        img_keras = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        arr = np.asarray(img_keras, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        t0 = time.time()
        preds = models["keras"].predict(arr, verbose=0)
        t1 = time.time()
        idx_k = int(np.argmax(preds))
        conf_k = float(np.max(preds))
        name_k = CLASS_NAMES[idx_k] if idx_k < len(CLASS_NAMES) else "Unknown"
        time_k = (t1 - t0) * 1000

        # ── PyTorch prediction ────────────────────────────────────────
        tf_pytorch = transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        tensor = tf_pytorch(image).unsqueeze(0)
        t0 = time.time()
        with torch.no_grad():
            out = models["pytorch"](tensor)
            probs = torch.nn.functional.softmax(out[0], dim=0)
        t1 = time.time()
        conf_pt, idx_pt = torch.max(probs, 0)
        name_pt = CLASS_NAMES[idx_pt.item()] if idx_pt.item() < len(CLASS_NAMES) else "Unknown"
        time_pt = (t1 - t0) * 1000

        info_k = get_info(name_k)
        info_pt = get_info(name_pt)

        return {
            "keras": {"name": name_k, "confidence": conf_k, "latency_ms": time_k, "info": info_k},
            "pytorch": {"name": name_pt, "confidence": conf_pt.item(), "latency_ms": time_pt, "info": info_pt},
            "consensus": name_k == name_pt,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
def chat(req: ChatRequest):
    import requests as http_requests
    try:
        context = """You are an expert Agricultural Consultant for 'Agriconnect', specializing in Pomegranate and Mango farming.
Knowledge Base:
1. Crops: Pomegranate (Bhagwa, Arakta) and Mango (Kesar, Alphonso, Dasheri).
2. Diseases: Alternaria, Anthracnose, Bacterial Blight, Cercospora, Powdery Mildew.
3. Treatments: Recommend Organic (Neem oil) and Chemical (Mancozeb) remedies with dosage.
Guidelines: Be concise, practical, and farmer-friendly. Use bullet points."""
        full_prompt = f"{context}\n\nUser: {req.message}"

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        payload = {"contents": [{"parts": [{"text": full_prompt}]}]}
        r = http_requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        reply = data["candidates"][0]["content"]["parts"][0]["text"]
        return {"reply": reply}
    except Exception as e:
        return {"reply": f"Sorry, I couldn't connect to the AI assistant. Error: {str(e)}"}


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=False)
