@echo off
chcp 65001 >nul
echo ================================
echo Gray Extractor Simple
echo ================================
echo Activating conda qt6...
call "E:\anaconda3\condabin\conda.bat" activate qt6
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate conda qt6 environment
    pause
    exit /b 1
)
echo OK.
echo Starting Simple Gray Extractor...
python "E:\Hermes Folder\projects\manual-thickness\gray_extractor_Simple\gray_extractor_simple.py"
if %errorlevel% neq 0 (
    echo [ERROR] Script exited with code %errorlevel%
    pause
)
