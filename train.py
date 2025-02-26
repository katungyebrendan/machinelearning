import torch
import torch.nn.functional as F
from model import TeacherGNN, StudentGNN

# Define model parameters
INPUT_SIZE = 5   # Adjust based on your dataset
HIDDEN_SIZE = 64
OUTPUT_SIZE = 2  # Adjust based on the number of classes

# Initialize models
teacher_model = TeacherGNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
student_model = StudentGNN(INPUT_SIZE, 32, OUTPUT_SIZE)

# Define optimizer
optimizer_teacher = torch.optim.Adam(teacher_model.parameters(), lr=0.01)
optimizer_student = torch.optim.Adam(student_model.parameters(), lr=0.01)

# Dummy dataset
x = torch.randn((10, INPUT_SIZE), requires_grad=True)  # 10 samples, 5 features
edge_index = torch.randint(0, 10, (2, 20))  # 20 edges
y = torch.randint(0, OUTPUT_SIZE, (10,))  # Random labels

# 🔹 Training function
def train(model, optimizer):
    model.train()
    optimizer.zero_grad()
    
    output = model(x, edge_index)  # Forward pass
    loss = F.nll_loss(output, y)   # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step() # Update weights
    return loss.item()

# Train both models for a few epochs
for epoch in range(5):
    loss_teacher = train(teacher_model, optimizer_teacher)
    loss_student = train(student_model, optimizer_student)
    print(f"Epoch {epoch+1}: Teacher Loss = {loss_teacher:.4f}, Student Loss = {loss_student:.4f}")

# 🔹 Save models
torch.save(teacher_model.state_dict(), "teacher_model.pth")
torch.save(student_model.state_dict(), "student_model.pth")
print("✅ Models trained and saved successfully!")
