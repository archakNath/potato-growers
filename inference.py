import onnxruntime as ort
import numpy as np
from PIL import Image

# Load the ONNX model
ort_session = ort.InferenceSession("plant_disease_model.onnx")

# Define the image transformation
def preprocess_image(image_path):
    image = Image.open(image_path).resize((256, 256))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # Change HWC to CHW
    image = np.expand_dims(image, axis=0)
    return image

def predict(image_path):
    image = preprocess_image(image_path)
    
    ort_inputs = {ort_session.get_inputs()[0].name: image}
    ort_outs = ort_session.run(None, ort_inputs)
    
    prediction = np.argmax(ort_outs[0])
    
    return prediction

# Test the prediction function
image_path = "testImage.jpg"
class_names = 6 ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Leaf_Mold', 'Tomato_healthy']
  # Replace with your class names
prediction = predict(image_path)
print(f"Predicted class: {class_names[prediction]}")
