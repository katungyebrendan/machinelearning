import torch
from model import TeacherGNN, StudentGNN
# Define model parameters (must match training)
INPUT_SIZE = 5   # Adjust based on dataset
HIDDEN_SIZE = 64
OUTPUT_SIZE = 2  # Adjust based on number of classes
# Paths to saved models
teacher_model_path = "teacher_model.pth"
student_model_path = "student_model.pth"
# Initialize models
teacher_model = TeacherGNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
student_model = StudentGNN(INPUT_SIZE, 32, OUTPUT_SIZE)
# Load saved weights
teacher_model.load_state_dict(torch.load(teacher_model_path))
student_model.load_state_dict(torch.load(student_model_path))
# Set models to evaluation mode
teacher_model.eval()
student_model.eval()
print("✅ Models loaded successfully!")