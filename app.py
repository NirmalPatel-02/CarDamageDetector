from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import base64

app = FastAPI(title="CarGuard AI - Damage Detection System")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

os.makedirs("uploads", exist_ok=True)

print("Loading AI models...")
car_model = tf.keras.models.load_model('models/car_or_not_predictor.keras')
damage_model = tf.keras.models.load_model('models/car_damage_detection.keras')
severity_model = tf.keras.models.load_model('models/how_much_damage_model.keras')

from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess

first_load = True

def prepare_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img)
    arr = np.expand_dims(arr, axis=0)
    return arr

def image_to_base64(img_path: str) -> str:
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    global first_load
    contents = await file.read()
    filename = f"uploads/{file.filename}"
    with open(filename, "wb") as f:
        f.write(contents)

    try:
        img = Image.open(filename)
        x = prepare_image(img)
        x1 = eff_preprocess(x.astype('float32'))

        is_first_load = first_load
        first_load = False  

        car_prob = float(car_model.predict(x1, verbose=0)[0][0])
        car_conf = (1 - car_prob) * 100

        if car_prob > 0.5:
            return {
                "result": "no_car",
                "confidence": f"{car_prob*100:.1f}%",
                "image": image_to_base64(filename),
                "is_first_load": is_first_load
            }

        good_prob = float(damage_model.predict(x1, verbose=0)[0][0])
        damage_conf = (1 - good_prob) * 100
        is_damaged = good_prob < 0.5

        if not is_damaged:
            overall_conf = (car_conf + (good_prob * 100)) / 2
            return {
                "result": "undamaged",
                "confidence": f"{overall_conf:.1f}%",
                "image": image_to_base64(filename),
                "is_first_load": is_first_load
            }

        pred = severity_model.predict(x1, verbose=0)[0]
        idx = np.argmax(pred)
        severity = ["Minor", "Moderate", "Severe"][idx]
        severity_conf = float(pred[idx]) * 100

        overall_conf = (car_conf + damage_conf + severity_conf) / 3

        severity_messages = {
            "Minor": "Just a few scratches – nothing serious. Easy fix!",
            "Moderate": "Visible damage detected. Needs panel repair or replacement.",
            "Severe": "Major structural damage found! Do not drive – safety risk!"
        }

        return {
            "result": "damaged",
            "severity": severity,
            "overall_confidence": f"{overall_conf:.1f}%",
            "message": severity_messages[severity],
            "image": image_to_base64(filename),
            "is_first_load": is_first_load
        }

    except Exception as e:
        print(f"Error: {e}")
        return {"result": "error", "message": "Failed to process image.", "is_first_load": False}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)