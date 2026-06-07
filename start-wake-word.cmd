@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found: .venv\Scripts\python.exe
    echo Create it first, then install dependencies from requirements.txt.
    exit /b 1
)

".venv\Scripts\python.exe" -m wake_word_detector --config config.yaml %*
exit /b %ERRORLEVEL%
