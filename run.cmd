@echo off
echo ========================================================
echo               VoiceGuard AI Launcher
echo ========================================================
echo.
echo Starting VoiceGuard AI Frontend...
cd frontend
if not exist node_modules (
    echo Installing frontend dependencies (one-time setup)...
    call npm install
)
call npm run dev
