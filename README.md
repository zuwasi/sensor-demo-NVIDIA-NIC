# CUDA Profiler GUI

A professional Python-based GUI application for profiling CUDA applications using NVIDIA Nsight Systems, with built-in performance comparison and visualization.

![CUDA Profiling](https://img.shields.io/badge/CUDA-13.0-green.svg)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Features

✨ **One-Click CUDA Profiling**
- Profile any CUDA executable with a single click
- Real-time output display with color-coded messages
- Automatic metrics collection and storage

📊 **Performance Comparison**
- Compare multiple profiling runs side-by-side
- Interactive bar charts with trend lines
- Track performance improvements over time

📈 **Multiple Comparison Views**
- **Kernel Times**: Compare individual CUDA kernel performance
- **Memory Operations**: Analyze memory transfer bottlenecks
- **CUDA API Calls**: Identify API overhead
- **Total Time**: Overall execution time trends

🎯 **Smart Analytics**
- Automatic performance trend analysis
- Percentage improvement calculations
- Historical metrics tracking in JSON format

## Screenshots

### Main Profiler Window
- Select CUDA executable
- Configure output directory and report names
- View real-time profiling output

### Comparison Window
- Select multiple runs to compare
- Choose comparison type (kernels, memory, API, total time)
- Interactive charts with zoom and pan

## Requirements

### Runtime Requirements
- Windows 10/11
- NVIDIA GPU with CUDA support
- NVIDIA Driver 581.57+ (for CUDA 13.0)
- CUDA Toolkit 13.0+
- NVIDIA Nsight Systems 2025.3.2+

### Python Dependencies
```bash
pip install matplotlib numpy
```

Or run:
```cmd
install_profiler_dependencies.bat
```

## Installation

### Option 1: Run from Source
```cmd
git clone <repository-url>
cd <repository-folder>
install_profiler_dependencies.bat
python cuda_profiler_gui.py
```

### Option 2: Build Standalone Executable
```cmd
build_profiler_exe.bat
```
Creates: `dist\CUDA_Profiler_GUI.exe` (standalone, no Python needed)

## Usage

### Basic Profiling

1. Launch the application:
   ```cmd
   python cuda_profiler_gui.py
   ```

2. Click **"Browse..."** next to "CUDA Executable"
3. Select your compiled CUDA application
4. Click **"▶ Profile Application"**
5. View real-time profiling results

### Comparing Runs

1. Profile your application multiple times (e.g., before and after optimization)
2. Click **"📈 Compare Reports"**
3. Select 2+ runs using Ctrl+Click
4. Choose a comparison type:
   - **📊 Compare Kernel Times** - See individual kernel performance
   - **💾 Compare Memory Ops** - Analyze memory bottlenecks
   - **⚡ Compare CUDA API** - Check API call overhead
   - **📈 Compare Total Time** - Overall execution trends

### Understanding Results

- **Green text**: Success messages and key metrics
- **Orange text**: Warnings (informational, usually safe to ignore)
- **Red text**: Errors
- **Trend line**: Shows performance direction (upward = slower, downward = faster)
- **Change %**: Improvement/regression from first to last run

## Example: CUDA Runtime Demo

Included in `source/repos/cuda_runtime_demo/`:

### Original Version
```cuda
// Basic kernel
__global__ void add1(float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += 1.0f;
}
```

### Optimized Version
Features:
- Grid-stride loops for better scalability
- Vectorized memory access (float4)
- Optimized thread configurations
- Multiple test configurations

**Expected speedup: 2-4x** (from vectorization)

Build optimized version:
```cmd
cd source/repos/cuda_runtime_demo/cuda_runtime_demo
build_optimized.bat
```

## Project Structure

```
.
├── cuda_profiler_gui.py              # Main GUI application
├── cuda_profiler_gui_admin.bat       # Launch with admin privileges
├── install_profiler_dependencies.bat # Install Python dependencies
├── build_profiler_exe.bat            # Build standalone executable
├── build_profiler_folder.bat         # Build folder distribution
├── PACKAGING_GUIDE.md                # Distribution guide
├── source/
│   └── repos/
│       └── cuda_runtime_demo/
│           ├── cuda_runtime_demo/
│           │   ├── main.cu                    # Original CUDA code
│           │   ├── main_optimized.cu          # Optimized version
│           │   └── build_optimized.bat        # Build script
│           └── OPTIMIZATION_GUIDE.md          # Optimization explanations
└── README.md
```

## Metrics Tracked

The profiler automatically tracks:
- **CUDA API Calls**: cudaMalloc, cudaMemcpy, cudaLaunchKernel, etc.
- **Kernel Execution Times**: Individual GPU kernel performance
- **Memory Operations**: Device-to-Host, Host-to-Device transfers
- **Total Execution Time**: End-to-end profiling metrics

All metrics saved to: `Documents\cuda_profiling_metrics.json`

## Optimization Tips

1. **Profile first**: Always measure before optimizing
2. **Compare incrementally**: Make one change at a time
3. **Focus on hotspots**: Optimize what the profiler shows is slow
4. **Memory is key**: Most CUDA apps are memory-bound, not compute-bound
5. **Vectorize**: Use float4, int4 for better memory coalescing

See `OPTIMIZATION_GUIDE.md` for detailed techniques.

## Troubleshooting

### "No CUDA device found"
- Check GPU drivers: `nvidia-smi`
- Verify CUDA Toolkit installation
- Ensure CUDA version matches driver

### "Nsight Systems not found"
- Install Nsight Systems 2025.3.2+
- Or update path in `cuda_profiler_gui.py` line 19

### "Permission denied" / ERR_NVGPUCTRPERM
- RTX 5080 laptop GPUs block low-level profiling
- Use Nsight Systems (timeline) instead of Nsight Compute (counters)
- This profiler uses Nsight Systems (works on all GPUs)

### Charts not showing
```cmd
pip install --upgrade matplotlib numpy
```

## Development

### Running Tests
Profile the included demo:
```cmd
python cuda_profiler_gui.py
# Select: source/repos/cuda_runtime_demo/x64/Debug/cuda_runtime_demo.exe
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - feel free to use in your projects!

## Acknowledgments

- Built with Python, tkinter, matplotlib, numpy
- Powered by NVIDIA Nsight Systems
- Tested on RTX 5080 Laptop GPU (Compute Capability 12.0)

## Author

Created for CUDA performance optimization and profiling workflows.

---

**Happy Profiling! 🚀**
