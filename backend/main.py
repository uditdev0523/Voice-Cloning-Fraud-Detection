from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Form, Request
from fastapi.responses import JSONResponse, FileResponse
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
import soundfile as sf
import traceback
from database import SessionLocal, Transaction
from models import CRNNWithAttn

app = FastAPI(title="VoiceGuard AI API")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    print(f"Exception on {request.url}: {tb}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {str(exc)}", "traceback": tb}
    )

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

def load_audio_tensor(file_path: str):
    # 1. Try soundfile directly
    try:
        data, sr = sf.read(file_path, dtype='float32')
        if data.ndim == 1:
            tensor = torch.from_numpy(data).unsqueeze(0)
        else:
            tensor = torch.from_numpy(data.T)
        return tensor, sr
    except Exception:
        pass

    # 2. Try torchaudio directly
    try:
        waveform, sr = torchaudio.load(file_path)
        return waveform, sr
    except Exception:
        pass

    # 3. Use ffmpeg to transcode to standard 16kHz 2-channel 16-bit PCM WAV
    converted_path = file_path + ".converted.wav"
    try:
        cmd = ["ffmpeg", "-y", "-i", file_path, "-vn", "-ar", "16000", "-ac", "2", "-c:a", "pcm_s16le", converted_path]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if os.path.exists(converted_path) and os.path.getsize(converted_path) > 0:
            data, sr = sf.read(converted_path, dtype='float32')
            if data.ndim == 1:
                tensor = torch.from_numpy(data).unsqueeze(0)
            else:
                tensor = torch.from_numpy(data.T)
            return tensor, sr
        else:
            err_msg = proc.stderr.decode(errors='ignore')[-300:]
            raise ValueError(f"ffmpeg conversion failed: {err_msg}")
    finally:
        if os.path.exists(converted_path):
            try:
                os.remove(converted_path)
            except Exception:
                pass

def preprocess(waveform, sample_rate):
    if sample_rate != 16000:
        resample = T.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resample(waveform)

    # Ensure 2 channels
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2, :]

    max_len = 16000 * 4
    all_waveforms = []
    total_len = waveform.shape[1]
    
    if total_len == 0:
        return []

    # Slice into uniform 4-second chunks
    start = 0
    while start < total_len:
        end = start + max_len
        chunk = waveform[:, start:end]
        if chunk.shape[1] < max_len:
            pad_len = max_len - chunk.shape[1]
            chunk = torch.nn.functional.pad(chunk, (0, pad_len))
        all_waveforms.append(chunk)
        start = end

    all_spec = []
    for wave in all_waveforms:
        spec = amp_to_db(mel_transform(wave))
        all_spec.append(spec)
    return all_spec

@app.post("/api/inference")
async def inference(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name

    try:
        waveform, sample_rate = load_audio_tensor(tmp_file_path)
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
        return {"prediction": label, "confidence": float(confidence)}
    finally:
        if os.path.exists(tmp_file_path):
            try:
                os.remove(tmp_file_path)
            except Exception:
                pass

@app.post("/api/transactions/verify")
async def verify_transaction(
    amount: float = Form(...),
    recipient: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Predict voice authenticity
    inference_result = await inference(file)
    confidence = float(inference_result.get("confidence", 0.5))
    label = str(inference_result.get("prediction", "Uncertain"))
    
    # Risk Engine Logic
    if label == "Fake":
        risk_score = 1.0 - (confidence * 0.5) # Minimum 50% risk for fake
    else:
        risk_score = 1.0 - confidence
        
    risk_score = min(max(float(risk_score), 0.0), 1.0)
    
    status = "approved"
    if risk_score > 0.6 or label == "Fake":
        status = "declined"
    elif risk_score > 0.3 and float(amount) > 1000:
        status = "hold"
        
    # Save Transaction
    tx = Transaction(amount=float(amount), recipient=str(recipient), status=status, risk_score=float(risk_score))
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


