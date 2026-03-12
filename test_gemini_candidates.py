
import google.generativeai as genai
import time

GEMINI_API_KEY = "AIzaSyBgndLh6fx_GIsfyFX4Zj335EFDR_iHKaw"
genai.configure(api_key=GEMINI_API_KEY)

candidates = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-preview-02-05",
    "gemini-flash-latest",
    "gemini-1.5-flash-001"
]

print("Testing models...")

for model_name in candidates:
    print(f"\n--- Testing {model_name} ---")
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Hello, strictly one word answer.")
        print(f"SUCCESS: {response.text}")
        # If success, verify vision support (optional, but good to know)
        if 'image' in model_name or 'flash' in model_name: 
             print("Likely supports vision.")
        break # Stop at first success to save quota/time
    except Exception as e:
        print(f"FAILED: {e}")
        time.sleep(1)
