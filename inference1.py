import torch
import torchvision.transforms as transforms
from PIL import Image
from model import CNNModel  # Import your model class from model.py

# Load the state dict (replace 'path_to_your_model.pth' with your actual path)
state_dict = torch.load("plant_disease_model.pth", map_location=torch.device('cpu'))

# Instantiate your model
model = CNNModel()
model.load_state_dict(state_dict)
model.eval()

# Define image transformations (if required)
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the image
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Function for model inference
def predict_image(image_path):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image_tensor)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()  # Assuming single image prediction

# Example usage:
image_path = "/testImage.jpg"
prediction = predict_image(image_path)
print(f"Predicted class index: {prediction}")
