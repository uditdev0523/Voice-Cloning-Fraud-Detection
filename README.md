# VoiceGuard AI — Real-Time Voice Cloning Fraud Detection

VoiceGuard AI is a sophisticated, real-time system designed to identify potential voice-cloning fraud in banking and payment workflows. It operates by analyzing voice inputs during transaction verifications and providing a rolling risk score based on the authenticity of the voice.

This project was built upon the [Fraud Audio Detection](https://github.com/mlvanguards/fraud-audio-detection) repository, reusing the powerful ResNet18 + Bi-GRU deep learning architecture. 

## Features

- **Real-Time Inference API:** FastAPI-powered backend for fast and efficient audio fraud detection.
- **Transaction Simulator:** A comprehensive banking dashboard to simulate high-risk money transfers.
- **Microphone Audio Processing:** Records voice data directly from the browser for live evaluation.
- **Privacy & Security First:** Adheres to consent-driven recording, short-lived audio buffers, minimal metadata logging, and avoids keeping raw recordings in application logs.
- **Rolling Risk Engine:** Evaluates transactions not just on voice cloning probability, but combined with the transaction amount and user risk context.

## Technology Stack

- **Backend:** FastAPI, Python, PyTorch, SQLAlchemy (SQLite), Uvicorn
- **Frontend:** React, Vite, TailwindCSS, Lucide Icons
- **Machine Learning:** PyTorch, TorchAudio (ResNet18 + Bi-GRU)
- **Deployment:** Docker

## Installation (Windows PowerShell)

Ensure you have Python 3.11+, Node.js 18+, and ffmpeg installed on your Windows machine.

### 1. Clone the repository and set up Backend
```powershell
git clone https://github.com/your-username/voiceguard-ai.git
cd voiceguard-ai\backend

# Create virtual environment and install dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run the FastAPI server
uvicorn main:app --reload
```

### 2. Set up Frontend
Open a new PowerShell window:
```powershell
cd voiceguard-ai\frontend

# Install node dependencies
npm install

# Run the Vite development server
npm run dev
```

### 3. Environment Variables
Copy `.env.example` to `.env` in the root directory and configure as needed.
By default, the backend runs on `http://localhost:8000`. No external API keys are required for core functionality.

## API Documentation

- `POST /api/inference`
  - Accepts a `multipart/form-data` with a `file` field containing audio (`.wav`, `.flac`, `.webm`).
  - Returns: `{"prediction": "Real", "confidence": 0.95}`

- `POST /api/transactions/verify`
  - Accepts `amount`, `recipient`, and `file` via form data.
  - Simulates a banking transaction workflow and risk scoring based on voice input.
  - Returns transaction details, status (`approved`, `declined`, `hold`), and risk metrics.

- `GET /api/transactions`
  - Retrieves a list of past simulated transactions.

## Evaluation and Testing

Evaluation is a crucial part of VoiceGuard AI. To run the automated tests on the FastAPI backend:
```powershell
cd backend
pytest test_main.py
```
*Note: Due to lack of localized evaluation datasets, full F1 Score, Precision, and Recall testing matrices are omitted from unit tests but can be run using the original model scripts.*

## Docker Deployment
Build and run the backend using Docker:
```powershell
docker build -t voiceguard-api .
docker run -p 8000:8000 voiceguard-api
```

## Security & Privacy
- **Consent:** The frontend explicitly asks for microphone permission and only records when the user presses the 'Record' button.
- **Data Minimization:** Audio is kept in memory during inference and temporary files are immediately deleted. No raw audio is stored in the SQLite database.
- **Attribution:** Model architecture and weights are attributed to MLVanguards.

## Limitations & Troubleshooting
- **Model Bias:** The underlying model may misclassify heavily compressed telephone audio or thick accents. 
- **Dependencies:** If you face torchaudio/webm issues, ensure `ffmpeg` is installed and in your system PATH.