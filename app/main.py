# main.py
from fastapi import FastAPI, UploadFile, File, Depends
from sqlalchemy.orm import Session
from .inference import predict
from fastapi.middleware.cors import CORSMiddleware
import logging
from fastapi.staticfiles import StaticFiles
import os

from .connectionDb import get_db
from .models import Inference


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

app = FastAPI(title="Ganoderma Detection API")
app.mount("/static", StaticFiles(directory=".", html=False), name="static")

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://0.0.0.0:5173",
    "http://192.168.1.42:5173",
    "http://192.168.1.9:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# @app.post("/predict")
# async def predict_image(file: UploadFile = File(...)):
#     image_bytes = await file.read()
#     result = predict(image_bytes)

#     print("DEBUG raw result:", result)
#     return result


@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):

    image_bytes = await file.read()
    result = predict(image_bytes)

    new_inference = Inference(
        user_id=1,
        label=result["label"],
        confidence=result["confidence"],
        inference_time=result["inference_time"],
        gradcam_image=result["gradcam_image"]
    )

    db.add(new_inference)
    db.commit()
    db.refresh(new_inference)

    return result