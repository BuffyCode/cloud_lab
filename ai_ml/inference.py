import joblib
import json
import os
import numpy as np

def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))

def input_fn(request_body, request_content_type):
    data = json.loads(request_body)
    return np.array(data["input"]).reshape(1, -1)

def predict_fn(input_data, model):
    return model.predict(input_data).tolist()

def output_fn(prediction, content_type):
    return json.dumps(prediction)