from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
from typing import Optional
from prometheus_fastapi_instrumentator import Instrumentator


app = FastAPI(
    title="IndoDisaster MLOps API",
    description="API untuk prediksi status keadaan darurat bencana di Indonesia",
    version="1.0.0"
)


# Prometheus instrumentation
Instrumentator().instrument(app).expose(app)


# Load model
model = joblib.load("models/final_model.joblib")


# Request schema
class PredictionRequest(BaseModel):
    jenis_bencana: str
    klasifikasi_bencana: str
    skala_bencana: str
    provinsi: str
    kabupaten: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    bulan_mulai: int
    kuartal_mulai: int
    jumlah_kejadian_sebelumnya: Optional[float] = None
    hari_sejak_kejadian_sebelumnya: Optional[float] = None
    ada_kejadian_sebelumnya: Optional[float] = None


@app.get("/")
def root():
    return {
        "message": "IndoDisaster MLOps API is running",
        "model": "Logistic Regression"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.post("/predict")
def predict(request: PredictionRequest):

    input_data = pd.DataFrame([{
        "jenis_bencana": request.jenis_bencana,
        "klasifikasi_bencana": request.klasifikasi_bencana,
        "skala_bencana": request.skala_bencana,
        "provinsi": request.provinsi,
        "kabupaten": request.kabupaten,
        "latitude": request.latitude,
        "longitude": request.longitude,
        "bulan_mulai": request.bulan_mulai,
        "kuartal_mulai": request.kuartal_mulai,
        "jumlah_kejadian_sebelumnya": request.jumlah_kejadian_sebelumnya,
        "hari_sejak_kejadian_sebelumnya": request.hari_sejak_kejadian_sebelumnya,
        "ada_kejadian_sebelumnya": request.ada_kejadian_sebelumnya
    }])

    prediction = model.predict(input_data)[0]

    return {
        "prediction": prediction
    }