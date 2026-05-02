from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np

app = FastAPI()

session = ort.InferenceSession("model.onnx")

input_name = session.get_inputs()[0].name

class InputData(BaseModel):
    values: list[float]

@app.post("/predict")
def predict(data: InputData):

    input_data = np.array(
        [data.values],
        dtype=np.float32
    )

    result = session.run(
        None,
        {input_name: input_data}
    )

    return {
        "prediction": result[0].tolist()
    }