from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from model import TeacherGNN, StudentGNN
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="GNN Model API",
    description="API for making predictions with GNN teacher and student models",
    version="1.0.0"
)

# Define model parameters (must match training)
INPUT_SIZE = 5   # Adjust based on dataset
HIDDEN_SIZE = 64
OUTPUT_SIZE = 2  # Adjust based on number of classes

# Paths to saved models
teacher_model_path = "teacher_model.pth"
student_model_path = "student_model.pth"

# Initialize models
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
    
    # Initialize models
    teacher_model = TeacherGNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    student_model = StudentGNN(INPUT_SIZE, 32, OUTPUT_SIZE)
    
    # Load saved weights
    try:
        teacher_model.load_state_dict(torch.load(teacher_model_path))
        student_model.load_state_dict(torch.load(student_model_path))
        
        # Set models to evaluation mode
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
    # Validate input
    if len(input_data.features) != INPUT_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"Input features must have length {INPUT_SIZE}, got {len(input_data.features)}"
        )
    
    # Convert input to tensor
    input_tensor = torch.tensor(input_data.features, dtype=torch.float32).unsqueeze(0)
    
    # Select model based on input
    model = teacher_model if input_data.use_teacher_model else student_model
    model_name = "teacher" if input_data.use_teacher_model else "student"
    
    # Make prediction
    try:
        with torch.no_grad():
            output = model(input_tensor)
            
            # Get prediction class and confidence
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            return {
                "prediction": predicted_class,
                "confidence": confidence,
                "model_used": model_name
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Performance comparison endpoint
@app.post("/compare")
def compare_models(input_data: InputData):
    # Validate input
    if len(input_data.features) != INPUT_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"Input features must have length {INPUT_SIZE}, got {len(input_data.features)}"
        )
    
    # Convert input to tensor
    input_tensor = torch.tensor(input_data.features, dtype=torch.float32).unsqueeze(0)
    
    # Make predictions with both models
    try:
        with torch.no_grad():
            # Teacher model prediction
            teacher_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            teacher_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if torch.cuda.is_available():
                teacher_start.record()
            
            teacher_output = teacher_model(input_tensor)
            teacher_probs = torch.nn.functional.softmax(teacher_output, dim=1)
            teacher_pred = torch.argmax(teacher_output, dim=1).item()
            teacher_conf = teacher_probs[0][teacher_pred].item()
            
            if torch.cuda.is_available():
                teacher_end.record()
                torch.cuda.synchronize()
                teacher_time = teacher_start.elapsed_time(teacher_end)
            else:
                teacher_time = None
            
            # Student model prediction
            student_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            student_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if torch.cuda.is_available():
                student_start.record()
            
            student_output = student_model(input_tensor)
            student_probs = torch.nn.functional.softmax(student_output, dim=1)
            student_pred = torch.argmax(student_output, dim=1).item()
            student_conf = student_probs[0][student_pred].item()
            
            if torch.cuda.is_available():
                student_end.record()
                torch.cuda.synchronize()
                student_time = student_start.elapsed_time(student_end)
            else:
                student_time = None
            
            return {
                "teacher_model": {
                    "prediction": teacher_pred,
                    "confidence": teacher_conf,
                    "inference_time_ms": teacher_time
                },
                "student_model": {
                    "prediction": student_pred,
                    "confidence": student_conf,
                    "inference_time_ms": student_time
                },
                "agreement": teacher_pred == student_pred
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison error: {str(e)}")

# Run the server if this file is executed directly
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)