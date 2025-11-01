@echo off
echo ========================================
echo Building Optimized CUDA Application
echo ========================================
echo.

REM Set up Visual Studio compiler path
set "PATH=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64;%PATH%"

REM Set CUDA path
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"

echo Compiler: %CUDA_PATH%\bin\nvcc.exe
echo Target: RTX 5080 (sm_89)
echo Optimization: -O3 with fast math
echo.

REM Compile optimized version
echo Compiling main_optimized.cu...
"%CUDA_PATH%\bin\nvcc.exe" -O3 -std=c++17 --gpu-architecture=sm_89 --use_fast_math -o cuda_optimized.exe main_optimized.cu

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Build Successful!
    echo ========================================
    echo.
    echo Output: cuda_optimized.exe
    echo.
    echo Running optimized version...
    echo ========================================
    echo.
    cuda_optimized.exe
) else (
    echo.
    echo ========================================
    echo Build FAILED!
    echo ========================================
    echo Check errors above.
)

echo.
pause
