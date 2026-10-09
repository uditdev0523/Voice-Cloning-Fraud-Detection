from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
import datetime
from typing import List
import torch
import torchaudio
import torchaudio.transforms as T
import tempfile
import os
import subprocess
from database import SessionLocal, Transaction
from models import CRNNWithAttn

app = FastAPI(title="VoiceGuard AI API")

class TransactionResponse(BaseModel):
    id: int
    amount: float
    recipient: str
    status: str
    risk_score: float
    timestamp: datetime.datetime

    class Config:
        from_attributes = True
        orm_mode = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model without downloading redundant weights
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CRNNWithAttn(pretrained=False)
try:
    model.load_state_dict(torch.load('best_model10.pth', map_location=device))
    model.eval()
    model = model.to(device)
except Exception as e:
    print(f"Error loading model: {e}")

imagenet_mean = torch.tensor([0.485]).view(1, 1, 1, 1).to(device)
imagenet_std = torch.tensor([0.229]).view(1, 1, 1, 1).to(device)

# Pre-instantiate transforms once globally for optimal latency
mel_transform = T.MelSpectrogram(
    sample_rate=16000,
    n_fft=780,
    hop_length=195,
    n_mels=64
)
amp_to_db = T.AmplitudeToDB(top_db=80)

@app.on_event("startup")
async def startup_event():
    # Warm up model to eliminate cold inference latency on the first request
    try:
        dummy_tensor = torch.zeros((1, 2, 64, 329), device=device)
        with torch.no_grad():
            _ = model(dummy_tensor)
        print("Model warmed up successfully.")
    except Exception as e:
        print(f"Model warmup skipped: {e}")

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/api/health")
def api_health():
    return {"status": "ok", "service": "VoiceGuard AI API"}

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def preprocess(waveform, sample_rate):
    if sample_rate != 16000:
        resample = T.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resample(waveform)

    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)

    max_len = 16000 * 4
    all_waveforms = []
    for _ in range(waveform.shape[1] // max_len + 1):
        if waveform.shape[1] >= max_len:
            all_waveforms.append(waveform[:, :max_len])
            waveform = waveform[:, max_len:]
        elif waveform.shape[1] > 0 and waveform.shape[1] < max_len:
            pad_len = max_len - waveform.shape[1]
            all_waveforms.append(torch.nn.functional.pad(waveform, (0, pad_len)))

    all_spec = []
    for wave in all_waveforms:
        spec = amp_to_db(mel_transform(wave))
        all_spec.append(spec)
    return all_spec

@app.post("/api/inference")
async def inference(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name

    wav_path = tmp_file_path + ".wav"
    
    # Convert webm to wav if needed using torchaudio or ffmpeg
    try:
        waveform, sample_rate = torchaudio.load(tmp_file_path)
    except Exception as e:
        # Fallback to ffmpeg subprocess
        subprocess.run(["ffmpeg", "-y", "-i", tmp_file_path, wav_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(wav_path):
            waveform, sample_rate = torchaudio.load(wav_path)
        else:
            os.remove(tmp_file_path)
            raise HTTPException(status_code=400, detail=f"Error parsing audio: {e}")

    try:
        input_tensors = preprocess(waveform, sample_rate)
        
        if not input_tensors:
            return {"prediction": "Uncertain", "confidence": 0.5}

        # Vectorized batch inference for minimum latency
        batch_tensor = torch.stack(input_tensors).to(device)
        batch_tensor = (batch_tensor - imagenet_mean) / imagenet_std
        with torch.no_grad():
            outputs = model(batch_tensor)
            probs = torch.sigmoid(outputs)
            confidence = probs.mean().item()

        label = "Real" if confidence >= 0.4 else "Fake"
        return {"prediction": label, "confidence": confidence}
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)

@app.post("/api/transactions/verify")
async def verify_transaction(
    amount: float = Form(...),
    recipient: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Predict voice authenticity
    inference_result = await inference(file)
    confidence = inference_result["confidence"]
    label = inference_result["prediction"]
    
    # Risk Engine Logic
    # confidence > 0.4 means Real. If confidence is high, risk is low.
    # risk_score maps 0 to 1, where 1 is highest risk.
    if label == "Fake":
        risk_score = 1.0 - confidence # e.g. confidence = 0.2, risk = 0.8
    else:
        risk_score = 1.0 - confidence # confidence = 0.9, risk = 0.1
        
    # Scale risk slightly
    risk_score = min(max(risk_score, 0.0), 1.0)
    
    status = "approved"
    if risk_score > 0.7:
        status = "declined"
    elif risk_score > 0.4 and amount > 1000:
        status = "hold"
        
    # Save Transaction
    tx = Transaction(amount=amount, recipient=recipient, status=status, risk_score=risk_score)
    db.add(tx)
    db.commit()
    db.refresh(tx)
    
    return {
        "transaction_id": tx.id,
        "status": status,
        "risk_score": risk_score,
        "voice_prediction": label,
        "confidence": confidence
    }

@app.get("/api/transactions", response_model=List[TransactionResponse])
def get_transactions(db: Session = Depends(get_db)):
    return db.query(Transaction).order_by(Transaction.timestamp.desc()).limit(100).all()

# Serve static frontend directly
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

dist_dir = "dist" if os.path.exists("dist") else ("../frontend/dist" if os.path.exists("../frontend/dist") else None)

if dist_dir and os.path.exists(os.path.join(dist_dir, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(dist_dir, "assets")), name="assets")

@app.get("/favicon.svg")
def favicon():
    if dist_dir and os.path.exists(os.path.join(dist_dir, "favicon.svg")):
        return FileResponse(os.path.join(dist_dir, "favicon.svg"))
    raise HTTPException(status_code=404)

@app.get("/icons.svg")
def icons():
    if dist_dir and os.path.exists(os.path.join(dist_dir, "icons.svg")):
        return FileResponse(os.path.join(dist_dir, "icons.svg"))
    raise HTTPException(status_code=404)

@app.get("/")
def serve_index():
    if dist_dir and os.path.exists(os.path.join(dist_dir, "index.html")):
        return FileResponse(os.path.join(dist_dir, "index.html"))
    return {"status": "ok", "service": "VoiceGuard AI API"}

@app.get("/{full_path:path}")
def catch_all(full_path: str):
    if dist_dir:
        file_path = os.path.join(dist_dir, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        index_file = os.path.join(dist_dir, "index.html")
        if os.path.isfile(index_file):
            return FileResponse(index_file)
    raise HTTPException(status_code=404, detail="Not Found")


