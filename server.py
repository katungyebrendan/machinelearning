from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import uvicorn
from model import TeacherGNN, StudentGNN
import sys
import subprocess
import time

# Initialize FastAPI app
app = FastAPI(
    title="GNN Model API",
    description="API for making predictions with GNN teacher and student models",
    version="1.0.0"
)

# Define model parameters (must match training)
INPUT_SIZE = 5  # Adjust based on dataset
HIDDEN_SIZE = 64
OUTPUT_SIZE = 2  # Adjust based on number of classes

# Paths to saved models
teacher_model_path = "teacher_model.pth"
student_model_path = "student_model.pth"

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model = None
student_model = None

# Define input data model
class InputData(BaseModel):
    features: list
    use_teacher_model: bool = False

# Define response model
class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    model_used: str

# Load models at startup
@app.on_event("startup")
def load_models():
    global teacher_model, student_model
    try:
        teacher_model = TeacherGNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
        student_model = StudentGNN(INPUT_SIZE, 32, OUTPUT_SIZE).to(device)

        teacher_model.load_state_dict(torch.load(teacher_model_path, map_location=device))
        student_model.load_state_dict(torch.load(student_model_path, map_location=device))

        teacher_model.eval()
        student_model.eval()

        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        raise RuntimeError(f"Failed to load models: {str(e)}")

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "GNN Model API is running. Use /predict endpoint for predictions."}

# Health check endpoint
@app.get("/health")
def health_check():
    if teacher_model is None or student_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"status": "healthy", "models_loaded": True}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: InputData):
    if len(input_data.features) != INPUT_SIZE:
        raise HTTPException(status_code=400, detail=f"Input features must have length {INPUT_SIZE}, got {len(input_data.features)}")
    
    input_tensor = torch.tensor(input_data.features, dtype=torch.float32).unsqueeze(0).to(device)
    model = teacher_model if input_data.use_teacher_model else student_model
    model_name = "teacher" if input_data.use_teacher_model else "student"
    
    try:
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            return {"prediction": predicted_class, "confidence": confidence, "model_used": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Performance comparison endpoint
@app.post("/compare")
def compare_models(input_data: InputData):
    if len(input_data.features) != INPUT_SIZE:
        raise HTTPException(status_code=400, detail=f"Input features must have length {INPUT_SIZE}, got {len(input_data.features)}")
    
    input_tensor = torch.tensor(input_data.features, dtype=torch.float32).unsqueeze(0).to(device)
    try:
        with torch.no_grad():
            teacher_output = teacher_model(input_tensor)
            student_output = student_model(input_tensor)
            
            teacher_probs = torch.nn.functional.softmax(teacher_output, dim=1)
            student_probs = torch.nn.functional.softmax(student_output, dim=1)
            
            teacher_pred = torch.argmax(teacher_output, dim=1).item()
            student_pred = torch.argmax(student_output, dim=1).item()
            
            teacher_conf = teacher_probs[0][teacher_pred].item()
            student_conf = student_probs[0][student_pred].item()
            
            return {
                "teacher_model": {"prediction": teacher_pred, "confidence": teacher_conf},
                "student_model": {"prediction": student_pred, "confidence": student_conf},
                "agreement": teacher_pred == student_pred
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison error: {str(e)}")

# Start Uvicorn in a separate process if run inside Streamlit
if __name__ == "__main__":
    if "streamlit" in sys.modules:
        server_process = subprocess.Popen(["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"])
        time.sleep(3)
    else:
        uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
