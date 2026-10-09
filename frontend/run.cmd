@echo off
if not exist node_modules (
    echo [VoiceGuard AI] First run detected. Installing dependencies...
    call npm install
)
echo [VoiceGuard AI] Launching frontend development server...
call npm run dev
