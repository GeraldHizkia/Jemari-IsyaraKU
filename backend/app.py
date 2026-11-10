"""
Backend FastAPI Server untuk Jemari BISINDO Translator
Menyediakan REST API untuk inferensi pengenalan kata (WL-BISINDO),
daftar kosakata terorganisir, serta serving static files antarmuka web.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional

from inference import get_engine

app = FastAPI(
    title="Jemari BISINDO Translator API",
    description="Sistem Pengenal Bahasa Isyarat Indonesia (Alfabet + 32 Kosakata Kata WL-BISINDO)",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = get_engine()

class FrameFeature(BaseModel):
    features: List[float]  # 126 nilai fitur [L_x0, L_y0, ... R_z20]

class SequencePayload(BaseModel):
    sequence: List[List[float]] # List of 126D frames
    top_k: Optional[int] = 3

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "service": "Jemari BISINDO Translator",
        "version": "2.0.0",
        "word_classes_count": len(engine.labels),
        "device": str(engine.device)
    }

@app.get("/api/vocab")
def get_vocabulary():
    """Mengembalikan daftar 32 kosakata BISINDO beserta kategori dan label."""
    return {
        "count": len(engine.labels),
        "words": engine.labels
    }

@app.post("/api/predict/word")
def predict_word_sequence(payload: SequencePayload):
    """
    Menerima urutan fitur landmark temporal (T x 126) dan mengembalikan
    prediksi kata BISINDO dengan tingkat keyakinan (confidence).
    """
    if not payload.sequence:
        raise HTTPException(status_code=400, detail="Data sequence kosong.")

    result = engine.predict(payload.sequence, top_k=payload.top_k or 3)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return result

# Mount direktori web untuk antarmuka pengguna pada root '/'
web_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web"))
if os.path.exists(web_dir):
    app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
