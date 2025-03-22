from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import librosa
import tensorflow as tf
import uvicorn
from io import BytesIO

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load pre-trained model
MODEL_PATH = "model.h5"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Define emotion labels
emotions = ["Neutral", "Happy", "Sad", "Angry", "Fearful", "Disgusted", "Surprised", "Bored"]

# Function to extract MFCC features
def extract_features(audio, sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfccs = np.mean(mfccs, axis=1)
    return mfccs

# Function to predict emotion
def predict_emotion(audio, sr):
    features = extract_features(audio, sr)
    features = np.expand_dims(features, axis=0)  # Shape: (1, 40)
    features = np.expand_dims(features, axis=-1)  # Shape: (1, 40, 1)
    features = tf.convert_to_tensor(features, dtype=tf.float32)
    
    prediction = model.predict(features)
    emotion_index = np.argmax(prediction)
    return emotions[emotion_index]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        audio_buffer = BytesIO(audio_bytes)
        audio, sr = librosa.load(audio_buffer, sr=22050)
        
        if np.max(audio) < 0.01:  # Check if audio has significant amplitude
            return {"error": "No significant audio detected. Please check your microphone settings."}
        
        mood = predict_emotion(audio, sr)
        return {"emotion": mood}
    except Exception as e:
        return {"error": str(e)}

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
