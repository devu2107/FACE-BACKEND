from fastapi import FastAPI, File, UploadFile
import face_recognition
import numpy as np
import pickle
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


known_faces = []

# Load saved faces
try:
    with open("faces.pkl", "rb") as f:
        while True:
            known_faces.append(pickle.load(f))
except:
    pass

@app.get("/")
def home():
    return {"message": "Backend running"}

@app.post("/mark-attendance")
async def mark_attendance(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    
    image = face_recognition.load_image_file(file.file)
    encodings = face_recognition.face_encodings(image)

    if not encodings:
        return {"status": "No face detected"}

    for face in encodings:
        matches = face_recognition.compare_faces(known_faces, face)
        if True in matches:
            return {"status": "Present"}

    return {"status": "Unknown"}