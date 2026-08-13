@echo off
rem FoundationStereo Studio launcher (Windows).
rem Mirror of run_studio.sh: run from anywhere, uses the repo's own venv.
cd /d "%~dp0"

if not exist ".venv\Scripts\pythonw.exe" (
    echo No .venv found -- run:  python install.py
    echo   ^(it builds the environment, fetches the weights and verifies^)
    pause
    exit /b 1
)

rem pythonw = no console window; logs land in %TEMP%\fs_studio_*.log
".venv\Scripts\pythonw.exe" -m studio.app %*
