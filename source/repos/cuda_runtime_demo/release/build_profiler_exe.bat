@echo off
echo ========================================
echo Building CUDA Profiler GUI Executable
echo ========================================
echo.

REM Install dependencies if not already installed
echo Installing dependencies...
pip install -r ..\requirements-dev.txt

echo.
echo Building executable...
echo This will create a standalone .exe with all dependencies
echo.

REM Change to parent directory where cuda_profiler_gui.py is located
cd ..

REM Build single executable (all dependencies bundled)
python -m PyInstaller --onefile ^
    --windowed ^
    --name "CUDA_Profiler_GUI" ^
    --hidden-import matplotlib ^
    --hidden-import numpy ^
    --hidden-import tkinter ^
    cuda_profiler_gui.py

REM Move output to release folder
move /Y dist\CUDA_Profiler_GUI.exe release\CUDA_Profiler_GUI.exe

echo.
echo ========================================
echo Build Complete!
echo ========================================
echo.
echo Executable location: release\CUDA_Profiler_GUI.exe
echo.
echo You can distribute this single .exe file!
echo No Python installation required on target machine.
echo.
pause
