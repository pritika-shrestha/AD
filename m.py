import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from uuid import uuid4
from contextlib import asynccontextmanager

# FastAPI and dependencies
from fastapi import (
    FastAPI, 
    Request, 
    Form, 
    UploadFile, 
    File, 
    HTTPException,
    Depends, 
)
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates


# --- 1. CONFIGURATION AND INITIALIZATION ---

TEMPLATES = Jinja2Templates(directory="templates")

SCAN_UPLOAD_DIR = Path("uploads")
MODEL_DIR = Path("models")
SCAN_UPLOAD_DIR.mkdir(exist_ok=True) 

MODELS = {}

IN_PROGRESS_ASSESSMENTS: Dict[str, Dict[str, Any]] = {}


# --- 2. LIFESPAN EVENTS (Startup/Shutdown) ---

def load_models():
    """Synchronous function to load ML Models, called during startup."""
    print("Loading ML Models...")
    # ... (Model loading logic remains the same)
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield

app = FastAPI(title="NeuroPredict Assessment API", lifespan=lifespan)


# --- 3. CORE LOGIC FUNCTIONS ---

def get_assessment_data(session_id: str) -> Dict[str, Any]:
    """Fetches assessment data or raises a 404 error."""
    if session_id not in IN_PROGRESS_ASSESSMENTS:
        raise HTTPException(
            status_code=404, 
            detail="Assessment session not found or expired. Please restart assessment."
        )
    return IN_PROGRESS_ASSESSMENTS[session_id]

def perform_ml_prediction(assessment_data: Dict[str, Any]) -> tuple[str, float]:
    """Placeholder function for calling the loaded ML models."""
    if not MODELS:
        return "Low Risk (MOCK - Models Not Loaded)", 0.50
    cognitive_scores = assessment_data.get('cognitive_scores', {})
    total_score = cognitive_scores.get('total_score', 0)
    
    if total_score < 15:
        result = "High Risk: Probable Dementia"
        confidence = 0.92
    elif total_score < 30:
        result = "Moderate Risk: Mild Cognitive Impairment (MCI)"
        confidence = 0.78
    else:
        result = "Low Risk"
        confidence = 0.65
        
    return result, confidence


# --- 4. FASTAPI ROUTES ---

# --- HTML TEMPLATE ROUTES (Handles navigation and passes session_id) ---

@app.get("/", response_class=HTMLResponse, summary="Dashboard/Home Page", name="get_dashboard") # <-- FIX IS HERE
async def read_dashboard(request: Request):
    return TEMPLATES.TemplateResponse("dashboard.html", {"request": request})

@app.get("/assess/{step_id}", name="read_assessment_step", response_class=HTMLResponse, summary="Assessment Steps 1-5")
async def read_assessment_step(request: Request, step_id: int, session_id: Optional[str] = None):
    
    if not 1 <= step_id <= 5:
        raise HTTPException(status_code=404, detail="Invalid assessment step.")

    # Redirect to step 1 if session is missing for subsequent steps
    if step_id > 1 and not session_id:
        return RedirectResponse(url="/assess/1", status_code=303)
    
    # Check if the session ID provided is valid for steps > 1
    if step_id > 1 and session_id:
        try:
            get_assessment_data(session_id)
        except HTTPException:
            return RedirectResponse(url="/assess/1", status_code=303)


    context = {"request": request, "session_id": session_id}
    template_name = f"assess{step_id}.html"

    return TEMPLATES.TemplateResponse(template_name, context)

# -------------------------------------------------------------
# --- ASSESSMENT SUBMISSION ENDPOINTS (Corrected Patient Info) ---
# -------------------------------------------------------------

@app.post("/assess/patient-info", summary="Step 1: Save Patient Info and Start Session")
async def save_patient_info(
    full_name: str = Form(...), 
    dob: str = Form(...), 
    gender: str = Form(...), 
    contact: str = Form(...)
):
    """Creates a new session, saves patient data, and redirects to Step 2."""
    
    session_key = str(uuid4())
    
    IN_PROGRESS_ASSESSMENTS[session_key] = {
        "session_key": session_key,
        "patient_info": {
            "full_name": full_name, 
            "date_of_birth": dob, 
            "gender": gender, 
            "contact_number": contact
        }, 
        "medical_history": {},
        "cognitive_scores": {},
        "scan_storage_path": None,
    }
    
    return RedirectResponse(url=f"/assess/2?session_id={session_key}", status_code=303)


@app.post("/assess/medical-history/{session_id}", summary="Step 2: Save Medical History")
async def save_medical_history(
    session_id: str, 
    assessment_data: Dict[str, Any] = Depends(get_assessment_data), 
    hypertension: Optional[str] = Form(None), 
    diabetes: Optional[str] = Form(None), 
    stroke: Optional[str] = Form(None), 
    depression: Optional[str] = Form(None)
):
    """Stores Step 2 data temporarily and redirects to Step 3."""
    
    assessment_data["medical_history"] = {
        # Checkboxes are sent as "on" if checked, None otherwise
        "hypertension": hypertension == "on",
        "diabetes": diabetes == "on",
        "stroke": stroke == "on",
        "depression": depression == "on",
    }
    
    return RedirectResponse(url=f"/assess/3?session_id={session_id}", status_code=303)


@app.post("/assess/cognitive-scores/{session_id}", summary="Step 3: Save Cognitive Scores")
async def save_cognitive_scores(
    session_id: str,
    assessment_data: Dict[str, Any] = Depends(get_assessment_data), 
    orientation: int = Form(0), 
    memory: int = Form(0), 
):
    """Stores Step 3 data temporarily and redirects to Step 4."""
    
    total_score = orientation + memory
    
    assessment_data["cognitive_scores"] = {
        "orientation": orientation, 
        "memory": memory, 
        "total_score": total_score
    }
    
    return RedirectResponse(url=f"/assess/4?session_id={session_id}", status_code=303)


@app.post("/assess/scan-upload/{session_id}", summary="Step 4: Upload Medical Scan")
async def upload_scan(
    session_id: str, 
    assessment_data: Dict[str, Any] = Depends(get_assessment_data), 
    scan_file: UploadFile = File(..., alias="scan_file")
):
    """Saves the uploaded file locally and redirects to Step 5, with validation."""
    
    # ... (File validation and saving logic remains the same)
    
    # Mocking file saving for brevity
    if scan_file.filename:
        assessment_data["scan_storage_path"] = f"/uploads/{session_id}_{scan_file.filename}"
    
    return RedirectResponse(url=f"/assess/5?session_id={session_id}", status_code=303)


@app.post("/assess/submit/{session_id}", summary="Step 5: Final Submission & Prediction")
async def final_submit(
    session_id: str,
    assessment_data: Dict[str, Any] = Depends(get_assessment_data), 
):
    """Triggers ML prediction, stores final results in session, and redirects to results."""
    
    prediction_result, confidence_score = perform_ml_prediction(assessment_data)
    
    assessment_data["prediction_result"] = prediction_result
    assessment_data["confidence_score"] = confidence_score
    assessment_data["date_submitted"] = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

    # Correct redirection: Redirects to the final results page
    return RedirectResponse(url=f"/assessment/{session_id}/results", status_code=303)


@app.get("/assessment/{session_id}/results", response_class=HTMLResponse, summary="Step 6: View Results")
async def get_assessment_results(
    request: Request, 
    session_id: str,
    assessment: Dict[str, Any] = Depends(get_assessment_data) 
):
    """Loads all data from the temporary store for Step 6 and renders the results page (assess6.html)."""
    
    if "prediction_result" not in assessment:
        raise HTTPException(status_code=400, detail="Final submission not completed for this session.")

    risk_level = "success"
    if "High Risk" in assessment["prediction_result"]:
        risk_level = "danger"
    elif "Moderate Risk" in assessment["prediction_result"] or "MCI" in assessment["prediction_result"]:
        risk_level = "warning"
        
    context = {
        "request": request,
        "assessment_id": session_id,
        "patient_name": assessment['patient_info']['full_name'],
        "prediction_result": assessment["prediction_result"],
        "confidence_score": f"{assessment['confidence_score']:.0%}",
        "risk_level": risk_level,
        "date_submitted": assessment["date_submitted"],
    }
    
    return TEMPLATES.TemplateResponse("assess6.html", context)