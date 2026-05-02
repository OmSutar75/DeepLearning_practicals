import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model.onnx")

input_data = np.random.randn(1, 4).astype(np.float32)

outputs = session.run(
    None,
    {"input": input_data}
)

print(outputs)