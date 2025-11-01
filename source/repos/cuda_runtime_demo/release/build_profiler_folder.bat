@echo off
echo ========================================
echo Building CUDA Profiler GUI (Folder Distribution)
echo ========================================
echo.

REM Install dependencies if not already installed
echo Installing dependencies...
pip install -r ..\requirements-dev.txt

echo.
echo Building executable with DLLs in folder...
echo This creates faster startup than single-file exe
echo.

REM Change to parent directory where cuda_profiler_gui.py is located
cd ..

REM Build folder with exe + DLLs (faster startup, smaller size)
python -m PyInstaller --onedir ^
    --windowed ^
    --name "CUDA_Profiler_GUI" ^
    --hidden-import matplotlib ^
    --hidden-import numpy ^
    --hidden-import tkinter ^
    cuda_profiler_gui.py

REM Move output to release folder
if exist "release\CUDA_Profiler_GUI\" rmdir /S /Q "release\CUDA_Profiler_GUI\"
move /Y dist\CUDA_Profiler_GUI release\CUDA_Profiler_GUI

echo.
echo ========================================
echo Build Complete!
echo ========================================
echo.
echo Distribution folder: release\CUDA_Profiler_GUI\
echo Main executable: release\CUDA_Profiler_GUI\CUDA_Profiler_GUI.exe
echo.
echo Distribute the entire "CUDA_Profiler_GUI" folder
echo.
pause
