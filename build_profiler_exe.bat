@echo off
echo ========================================
echo Building CUDA Profiler GUI Executable
echo ========================================
echo.

REM Install PyInstaller if not already installed
echo Installing PyInstaller...
pip install pyinstaller

echo.
echo Building executable...
echo This will create a standalone .exe with all dependencies
echo.

REM Build single executable (all dependencies bundled)
pyinstaller --onefile ^
    --windowed ^
    --name "CUDA_Profiler_GUI" ^
    --icon="%SystemRoot%\System32\shell32.dll,16" ^
    --add-data "C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.3.2\target-windows-x64;nsight" ^
    --hidden-import matplotlib ^
    --hidden-import numpy ^
    --hidden-import tkinter ^
    cuda_profiler_gui.py

echo.
echo ========================================
echo Build Complete!
echo ========================================
echo.
echo Executable location: dist\CUDA_Profiler_GUI.exe
echo.
echo You can distribute this single .exe file!
echo No Python installation required on target machine.
echo.
pause
