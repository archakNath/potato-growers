import onnx
import onnxoptimizer

# Load your ONNX model
model_path = "plant_disease_model.onnx"
model = onnx.load(model_path)

# Apply optimization
optimized_model = onnxoptimizer.optimize(model)

# Save the optimized model
onnx.save(optimized_model, "plant_disease_model_optimized.onnx")
