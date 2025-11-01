@echo off
echo ========================================
echo Installing CUDA Profiler GUI Dependencies
echo ========================================
echo.

pip install -r ..\requirements.txt

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo You can now run the CUDA Profiler GUI:
echo python ..\cuda_profiler_gui.py
echo.
pause
