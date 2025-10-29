import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


model = load_model("pet_health_lstm_model.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
maxlen = 64  

class InputData(BaseModel):
    text: str

app = FastAPI()

@app.post("/predict")
def predict(data: InputData):
    seq = tokenizer.texts_to_sequences([data.text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=maxlen)
    probabilities = model.predict(padded) 
    top_class = np.argmax(probabilities[0])
    confidence = float(np.max(probabilities[0]))
    label = label_encoder.inverse_transform([top_class])[0]
    return {"prediction": label, "confidence": confidence}
