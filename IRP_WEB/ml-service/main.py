from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from model_utils import multimodal_predict

app = FastAPI(title="Multimodal Mental Health ML Service")


class PredictRequest(BaseModel):
    text: str
    image: Optional[str] = None
    cameraEnabled: bool = False
    session_id: Optional[str] = "default"


@app.get("/")
def root():
    return {"message": "ML service is running"}


@app.post("/predict")
def predict(req: PredictRequest):
    result = multimodal_predict(
        text=req.text,
        image_data=req.image,
        camera_enabled=req.cameraEnabled,
        session_id=req.session_id or "default"
    )
    return result