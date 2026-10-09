import React, { useState, useRef, useEffect } from 'react';
import { Mic, Square, Upload, CheckCircle, AlertTriangle, FileAudio, RefreshCw, Send, Wifi, WifiOff, Settings } from 'lucide-react';
import axios from 'axios';

function App() {
  const [activeTab, setActiveTab] = useState<'inference' | 'simulator'>('inference');
  const [recording, setRecording] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);

  // API Backend URL Configuration
  const defaultBase = typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
    ? 'http://localhost:8000'
    : (import.meta.env.VITE_API_URL || 'https://voice-cloning-fraud-detection.onrender.com');

  const [apiUrl, setApiUrl] = useState<string>(() => {
    return localStorage.getItem('voiceguard_api_url') || defaultBase;
  });
  const [showConfig, setShowConfig] = useState(false);
  const [backendStatus, setBackendStatus] = useState<'connected' | 'waking' | 'offline'>('waking');

  const [inferenceResult, setInferenceResult] = useState<{prediction: string, confidence: number} | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Simulator state
  const [amount, setAmount] = useState<string>('500');
  const [recipient, setRecipient] = useState<string>('John Doe');
  const [simResult, setSimResult] = useState<any>(null);

  const cleanUrl = apiUrl.replace(/\/+$/, '');

  // Pre-warm backend immediately on page load to eliminate cold-start latency
  useEffect(() => {
    let isMounted = true;
    const checkBackend = async () => {
      try {
        await axios.get(`${cleanUrl}/health`, { timeout: 8000 });
        if (isMounted) setBackendStatus('connected');
      } catch {
        if (isMounted) setBackendStatus('waking');
      }
    };

    checkBackend();
    const interval = setInterval(checkBackend, 10000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [cleanUrl]);

  const handleSaveApiUrl = (newUrl: string) => {
    setApiUrl(newUrl);
    localStorage.setItem('voiceguard_api_url', newUrl);
    setBackendStatus('waking');
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        const url = URL.createObjectURL(blob);
        setAudioUrl(url);
        const file = new File([blob], 'recording.webm', { type: 'audio/webm' });
        setAudioFile(file);
      };

      mediaRecorder.start();
      setRecording(true);
    } catch (err) {
      setError("Failed to access microphone. Please ensure permissions are granted.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      setRecording(false);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      setAudioFile(file);
      setAudioUrl(URL.createObjectURL(file));
      setInferenceResult(null);
      setSimResult(null);
    }
  };

  const submitInference = async () => {
    if (!audioFile) return;
    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', audioFile);

    try {
      const res = await axios.post(`${cleanUrl}/api/inference`, formData, { timeout: 60000 });
      setInferenceResult(res.data);
      setBackendStatus('connected');
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || "Failed to process audio. Ensure backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const submitSimulation = async () => {
    if (!audioFile) return;
    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', audioFile);
    formData.append('amount', amount);
    formData.append('recipient', recipient);

    try {
      const res = await axios.post(`${cleanUrl}/api/transactions/verify`, formData, { timeout: 60000 });
      setSimResult(res.data);
      setBackendStatus('connected');
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || "Failed to verify transaction. Ensure backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen p-8 bg-slate-900 font-sans text-slate-100 flex flex-col items-center">
      {/* Backend Latency / Status Banner */}
      <div className="w-full max-w-4xl mb-4 flex items-center justify-between px-4 py-2 bg-slate-800/80 rounded-lg border border-slate-700 text-xs text-slate-300">
        <div className="flex items-center space-x-2">
          {backendStatus === 'connected' ? (
            <span className="flex items-center text-emerald-400 font-medium">
              <span className="w-2 h-2 rounded-full bg-emerald-400 mr-2 animate-ping"></span>
              <Wifi className="w-3.5 h-3.5 mr-1" /> AI Engine Connected & Ready (Ultra-low latency)
            </span>
          ) : (
            <span className="flex items-center text-amber-400 font-medium">
              <span className="w-2 h-2 rounded-full bg-amber-400 mr-2 animate-pulse"></span>
              <WifiOff className="w-3.5 h-3.5 mr-1" /> Waking up AI engine in background (Render cold-start)...
            </span>
          )}
        </div>
        <button 
          onClick={() => setShowConfig(!showConfig)}
          className="flex items-center hover:text-indigo-400 transition-colors text-slate-400"
        >
          <Settings className="w-3.5 h-3.5 mr-1" /> API Settings
        </button>
      </div>

      {showConfig && (
        <div className="w-full max-w-4xl mb-4 p-4 bg-slate-800 rounded-lg border border-indigo-500/40 text-xs">
          <label className="block text-slate-300 font-medium mb-1">Backend API Base URL:</label>
          <div className="flex gap-2">
            <input 
              type="text" 
              value={apiUrl}
              onChange={(e) => handleSaveApiUrl(e.target.value)}
              className="flex-1 bg-slate-900 border border-slate-700 rounded px-3 py-1.5 text-slate-200 font-mono text-xs focus:outline-none focus:border-indigo-500"
              placeholder="e.g. https://your-app.onrender.com or http://localhost:8000"
            />
            <button 
              onClick={() => handleSaveApiUrl('http://localhost:8000')}
              className="px-3 py-1 bg-slate-700 hover:bg-slate-600 rounded text-slate-200"
            >
              Localhost
            </button>
          </div>
          <p className="text-slate-400 mt-1">Render Free Tier services take ~30-50s to wake up on the first request if idle.</p>
        </div>
      )}

      <header className="mb-8 text-center">
        <h1 className="text-4xl font-extrabold bg-gradient-to-r from-blue-400 to-indigo-500 bg-clip-text text-transparent mb-2">
          VoiceGuard AI
        </h1>
        <p className="text-slate-400 max-w-2xl text-lg">
          Real-time voice-cloning fraud detection for secure banking and payments.
        </p>
      </header>

      <div className="w-full max-w-4xl bg-slate-800 rounded-2xl shadow-xl overflow-hidden border border-slate-700">
        <div className="flex border-b border-slate-700">
          <button 
            className={`flex-1 py-4 text-center font-medium transition-colors ${activeTab === 'inference' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:bg-slate-700 hover:text-slate-200'}`}
            onClick={() => setActiveTab('inference')}
          >
            Voice Analysis
          </button>
          <button 
            className={`flex-1 py-4 text-center font-medium transition-colors ${activeTab === 'simulator' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:bg-slate-700 hover:text-slate-200'}`}
            onClick={() => setActiveTab('simulator')}
          >
            Banking Simulator
          </button>
        </div>

        <div className="p-8">
          {/* Audio Input Section (Shared) */}
          <div className="mb-8 p-6 bg-slate-900/50 rounded-xl border border-slate-700">
            <h3 className="text-lg font-semibold mb-4 text-slate-200 flex items-center">
              <FileAudio className="w-5 h-5 mr-2 text-indigo-400" /> Audio Input
            </h3>
            
            <div className="flex flex-wrap gap-4 items-center justify-center">
              {recording ? (
                <button onClick={stopRecording} className="flex items-center px-6 py-3 bg-red-500/20 text-red-400 rounded-lg border border-red-500/30 hover:bg-red-500/30 transition-all shadow-[0_0_15px_rgba(239,68,68,0.3)] animate-pulse">
                  <Square className="w-5 h-5 mr-2" /> Stop Recording
                </button>
              ) : (
                <button onClick={startRecording} className="flex items-center px-6 py-3 bg-indigo-500/20 text-indigo-400 rounded-lg border border-indigo-500/30 hover:bg-indigo-500/30 transition-all hover:shadow-[0_0_15px_rgba(99,102,241,0.2)]">
                  <Mic className="w-5 h-5 mr-2" /> Record Audio
                </button>
              )}
              
              <span className="text-slate-500 text-sm">OR</span>
              
              <label className="flex items-center px-6 py-3 bg-slate-800 text-slate-300 rounded-lg border border-slate-600 hover:bg-slate-700 cursor-pointer transition-all">
                <Upload className="w-5 h-5 mr-2" /> Upload File
                <input type="file" accept="audio/*" onChange={handleFileUpload} className="hidden" />
              </label>
            </div>

            {audioUrl && (
              <div className="mt-6 flex flex-col items-center">
                <audio src={audioUrl} controls className="w-full max-w-md h-12" />
              </div>
            )}
            
            {error && (
              <div className="mt-4 p-4 bg-red-500/10 border border-red-500/20 text-red-400 rounded-lg text-center">
                {error}
              </div>
            )}
          </div>

          {/* Tab Content */}
          {activeTab === 'inference' && (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
              <div className="flex justify-center">
                <button 
                  onClick={submitInference}
                  disabled={!audioFile || loading}
                  className="px-8 py-3 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-lg font-bold shadow-lg hover:shadow-indigo-500/25 disabled:opacity-50 disabled:cursor-not-allowed flex items-center transition-all"
                >
                  {loading ? <RefreshCw className="w-5 h-5 mr-2 animate-spin" /> : <CheckCircle className="w-5 h-5 mr-2" />}
                  Analyze Audio
                </button>
              </div>

              {inferenceResult && (
                <div className={`mt-8 p-6 rounded-xl border ${inferenceResult.prediction === 'Real' ? 'bg-emerald-500/10 border-emerald-500/20' : 'bg-rose-500/10 border-rose-500/20'}`}>
                   <div className="flex items-center justify-between">
                     <div>
                       <p className="text-sm text-slate-400 mb-1">Prediction</p>
                       <h3 className={`text-3xl font-bold flex items-center ${inferenceResult.prediction === 'Real' ? 'text-emerald-400' : 'text-rose-400'}`}>
                         {inferenceResult.prediction === 'Real' ? <CheckCircle className="w-8 h-8 mr-2" /> : <AlertTriangle className="w-8 h-8 mr-2" />}
                         {inferenceResult.prediction.toUpperCase()}
                       </h3>
                     </div>
                     <div className="text-right">
                       <p className="text-sm text-slate-400 mb-1">Confidence Score</p>
                       <p className="text-2xl font-mono text-slate-200">{(inferenceResult.confidence * 100).toFixed(2)}%</p>
                     </div>
                   </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'simulator' && (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div>
                  <label className="block text-sm font-medium text-slate-400 mb-2">Recipient Name</label>
                  <input 
                    type="text" 
                    value={recipient}
                    onChange={(e) => setRecipient(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-3 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all text-slate-200"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-400 mb-2">Transfer Amount ($)</label>
                  <input 
                    type="number" 
                    value={amount}
                    onChange={(e) => setAmount(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-3 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all text-slate-200"
                  />
                </div>
              </div>
              
              <div className="flex justify-center">
                <button 
                  onClick={submitSimulation}
                  disabled={!audioFile || loading}
                  className="px-8 py-3 bg-gradient-to-r from-emerald-500 to-teal-600 rounded-lg font-bold shadow-lg hover:shadow-emerald-500/25 disabled:opacity-50 disabled:cursor-not-allowed flex items-center transition-all"
                >
                  {loading ? <RefreshCw className="w-5 h-5 mr-2 animate-spin" /> : <Send className="w-5 h-5 mr-2" />}
                  Submit Transfer Request
                </button>
              </div>

              {simResult && (
                <div className={`mt-8 p-6 rounded-xl border ${simResult.status === 'approved' ? 'bg-emerald-500/10 border-emerald-500/20' : simResult.status === 'declined' ? 'bg-rose-500/10 border-rose-500/20' : 'bg-amber-500/10 border-amber-500/20'}`}>
                   <h4 className="text-xl font-bold mb-4 flex items-center">
                     Transaction {simResult.status.toUpperCase()}
                   </h4>
                   <div className="grid grid-cols-2 gap-4 text-sm">
                      <div className="bg-slate-900/50 p-3 rounded">
                        <span className="text-slate-500 block mb-1">Voice Prediction</span>
                        <span className={`font-medium ${simResult.voice_prediction === 'Real' ? 'text-emerald-400' : 'text-rose-400'}`}>{simResult.voice_prediction}</span>
                      </div>
                      <div className="bg-slate-900/50 p-3 rounded">
                        <span className="text-slate-500 block mb-1">Risk Score</span>
                        <span className="font-mono text-slate-200">{(simResult.risk_score * 100).toFixed(1)} / 100</span>
                      </div>
                   </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
