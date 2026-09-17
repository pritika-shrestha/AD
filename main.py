import os
import cv2
import numpy as np
import joblib
import tensorflow as tf
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from supabase import create_client, Client
import asyncio


# Patch to handle quantization_config from newer Keras versions
from keras.src.ops.operation import Operation

_original_from_config = Operation.from_config.__func__

@classmethod
def _patched_from_config(cls, config):
    config.pop('quantization_config', None)
    config.pop('optional', None)
    return _original_from_config(cls, config)

Operation.from_config = _patched_from_config

# 1. SUPABASE CONFIG
SUPABASE_URL = "https://svzrazwfbojkdshpudck.supabase.co"
SUPABASE_KEY = "YOUR_KEY"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 2. FASTAPI INIT
app = FastAPI(title="NeuroPredict Platform")
# templates = Jinja2Templates(directory="templates")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

templates = Jinja2Templates(
    directory=os.path.join(BASE_DIR, "templates")
)

# 3. CNN + SVM PIPELINE CLASS
class NeuroPredictor:
    def __init__(self, cnn_path, svm_path, scaler_path):
        # Load CNN model
        # self.cnn_model = tf.keras.models.load_model(cnn_path, compile=False)
        self.cnn_model = tf.keras.models.load_model(
    cnn_path,
    compile=False,
    safe_mode=False
)
        # Feature extractor: output of Dense(256) layer — layers[-3]
        # This matches exactly what was used during SVM training in the Colab notebook:
        #   feature_extractor = tf.keras.Model(
        #       inputs=cnn_model.layers[0].input,
        #       outputs=cnn_model.layers[-3].output   # Dense(256), before Dropout
        #   )
        self.feature_extractor = tf.keras.Model(
            inputs=self.cnn_model.layers[0].input,
            outputs=self.cnn_model.layers[-3].output
        )

        # Load SVM + scaler
        self.svm = joblib.load(svm_path)
        self.scaler = joblib.load(scaler_path)

        # Categories — MUST match sorted(os.listdir(train_dir)) used during training.
        # Alphabetical sort of the 4 class folder names produces this order:
        #   0: Mild Dementia
        #   1: Moderate Dementia
        #   2: Non Demented
        #   3: Very mild Dementia
        self.CATEGORIES = [
            'Mild Dementia',
            'Moderate Dementia',
            'Non Demented',
            'Very mild Dementia',
        ]

    def preprocess_image(self, file_path: str):
        """
        Load and preprocess a single MRI image to exactly match training:
          - Grayscale read
          - Resize to 224×224  (training used IMG_SIZE = 224, NOT 128)
          - float32 / 255.0
          - shape: (1, 224, 224, 1)
        """
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Image not found or unreadable: {file_path}")
        img = cv2.resize(img, (224, 224))          # ← must be 224, not 128
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=-1)         # (224, 224, 1)
        img = np.expand_dims(img, axis=0)          # (1, 224, 224, 1)
        return img

    def extract_features(self, img_path: str):
        img_tensor = self.preprocess_image(img_path)
        features = self.feature_extractor(img_tensor, training=False).numpy()  # (1, 256)
        return features

    def predict(self, img_path: str):

        img = self.preprocess_image(img_path)

        # ---------------- MRI VALIDATION ----------------
        img_2d = img[0, :, :, 0]

        # check grayscale distribution
        mean_val = img_2d.mean()
        std_val  = img_2d.std()

        # reject very bright or very dark images (like diagrams)
        if mean_val > 0.8 or mean_val < 0.05:
            return {
                "predicted_class": "INVALID MRI IMAGE",
                "confidence": 0.0,
                "probabilities": {},
                "debug_info": {"reason": "Not MRI brightness"}
            }

        # reject low texture images
        if std_val < 0.12:
            return {
                "predicted_class": "INVALID MRI IMAGE",
                "confidence": 0.0,
                "probabilities": {},
                "debug_info": {"reason": "Not MRI texture"}
            }

        # ---------------- FEATURE EXTRACTION ----------------
        features = self.feature_extractor(img, training=False).numpy()

        features_scaled = self.scaler.transform(features)

        pred_probs  = self.svm.predict_proba(features_scaled)[0]
        pred_class  = int(np.argmax(pred_probs))
        confidence  = float(pred_probs[pred_class])
        label       = self.CATEGORIES[pred_class]

        # reject low confidence
        if confidence < 0.75:
            return {
                "predicted_class": "INVALID MRI IMAGE",
                "confidence": confidence,
                "probabilities": {
                    self.CATEGORIES[i]: float(pred_probs[i])
                    for i in range(len(self.CATEGORIES))
                }
            }

        return {
            "predicted_class": label,
            "probabilities": {
                self.CATEGORIES[i]: float(pred_probs[i])
                for i in range(len(self.CATEGORIES))
            },
            "confidence": confidence,
        }
# 4. LOAD MODELS ON STARTUP
neuro_ai = NeuroPredictor(
    cnn_path='models/best_cnn_model.h5',
    svm_path='models/svm_calibrated.pkl',
    scaler_path='models/scaler.pkl'
)


# 5. API ENDPOINTS

# Home / dashboard
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/assess{num}.html", response_class=HTMLResponse)
async def serve_pages(request: Request, num: int):
    return templates.TemplateResponse(f"assess{num}.html", {"request": request})

@app.get("/dashboard.html", response_class=HTMLResponse)
async def dashboard_redirect(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

# Upload MRI scan
@app.post("/api/upload-scan")
async def upload_scan(mri_file: UploadFile = File(...)):
    try:
        content = await mri_file.read()
        file_path = os.path.join(UPLOAD_DIR, mri_file.filename)
        with open(file_path, "wb") as f:
            f.write(content)
        return JSONResponse({"status": "success", "filename": mri_file.filename})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# Combined upload + predict (used by assess4.html)
@app.post("/api/upload-and-predict")
async def upload_and_predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(content)
        result = neuro_ai.predict(file_path)
        return JSONResponse({
            "status": "success",
            "filename": file.filename,
            "result": result
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# Predict MRI scan
@app.post("/api/predict")
async def predict_scan(mri_filename: str = Form(...)):
    file_path = os.path.join(UPLOAD_DIR, mri_filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        result = neuro_ai.predict(file_path)
        return JSONResponse({"status": "success", **result})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# Submit assessment
@app.post("/api/submit-assessment")
async def submit_assessment(
    patientName: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    hand: str = Form(...),
    education: str = Form(...),
    mmse: float = Form(...),
    mri_filename: str = Form(...),
    prediction: str = Form(...)
):
    local_path = os.path.join(UPLOAD_DIR, mri_filename)
    if not os.path.exists(local_path):
        raise HTTPException(status_code=404, detail="MRI file not found")
    try:
        supabase_path = f"scans/{patientName.replace(' ', '_')}_{mri_filename}"

        def upload_file():
            with open(local_path, "rb") as f:
                return supabase.storage.from_("mri-scans").upload(
                    supabase_path,
                    f.read(),
                    file_options={"x-upsert": "true"}
                )

        await asyncio.to_thread(upload_file)

        def insert_db():
            return supabase.table("patients").insert({
                "patient_name": patientName,
                "age": age,
                "gender": gender,
                "hand": hand,
                "education_years": education,
                "mmse_score": mmse,
                "mri_scan_url": supabase_path,
                "prediction_result": prediction
            }).execute()

        res = await asyncio.to_thread(insert_db)

        return {"status": "success", "patient_id": res.data[0]["id"]}

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# Keep-alive ping
@app.get("/api/ping-db")
async def ping_db():
    try:
        def do_ping():
            return supabase.table("patients").select("id").limit(1).execute()
        await asyncio.to_thread(do_ping)
        return {"status": "ok", "message": "Supabase is awake"}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/api/debug-predict")
async def debug_predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(content)
        result = neuro_ai.predict(file_path)
        return JSONResponse({
            "status": "success",
            "prediction":    result.get("predicted_class"),
            "probabilities": result.get("probabilities"),
            "confidence":    result.get("confidence"),
            "debug":         result.get("debug_info", {}),
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# 6. RUN SERVER
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)