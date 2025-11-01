# CUDA Profiler GUI - Packaging Guide

## Two Distribution Options

### Option 1: Single EXE File (Easiest to distribute)
**Pros:**
- One file to distribute
- Users just double-click the .exe
- ~50-80 MB file size

**Cons:**
- Slower startup (extracts to temp folder each run)
- Larger file size

**Build:**
```cmd
build_profiler_exe.bat
```

**Output:** `dist\CUDA_Profiler_GUI.exe`

---

### Option 2: Folder with EXE + DLLs (Faster performance)
**Pros:**
- Faster startup time
- Smaller individual files
- Can update dependencies separately

**Cons:**
- Must distribute entire folder
- ~100-150 MB total

**Build:**
```cmd
build_profiler_folder.bat
```

**Output:** `dist\CUDA_Profiler_GUI\` folder with:
- `CUDA_Profiler_GUI.exe` (main executable)
- Multiple DLL files
- Python runtime files

---

## Requirements on Target Machine

### Your CUDA profiling apps will need:
1. ✅ **NVIDIA GPU** (CUDA-capable)
2. ✅ **NVIDIA Driver** (581.57 or newer)
3. ✅ **CUDA Toolkit 13.0** (or compatible version)
4. ✅ **Nsight Systems** (the packaged exe looks for it at standard location)

### The GUI executable includes:
- ✅ Python runtime (no Python installation needed)
- ✅ tkinter (GUI framework)
- ✅ matplotlib (charting)
- ✅ numpy (numerical operations)

---

## Building in VS 2022 (Alternative for C++ version)

If you want to rewrite the profiler in **C++** for VS 2022:

### Pros:
- Native performance
- No Python dependency
- Smaller executable (~2-5 MB)

### Cons:
- More code to write (GUI, charts, JSON parsing)
- Requires Qt or similar for GUI

### Steps:
1. Create new **C++ Windows Desktop Application** in VS 2022
2. Add **Qt** or **wxWidgets** for GUI framework
3. Use **Qt Charts** or **ChartDirector** for plotting
4. Link against **CUDA toolkit** libraries
5. Build as **Release** with **static linking**

**Static Linking in VS 2022:**
- Project → Properties → C/C++ → Code Generation → Runtime Library → **Multi-threaded (/MT)**
- This bundles the C++ runtime into the .exe

---

## Testing the Packaged Executable

### After building:

1. **Test on same machine:**
   ```cmd
   dist\CUDA_Profiler_GUI.exe
   ```

2. **Copy to different folder** (simulate clean environment)

3. **Test on different machine** with:
   - NVIDIA GPU
   - CUDA toolkit installed
   - Nsight Systems installed

---

## Troubleshooting

### "CUDA_PATH not found"
- Ensure CUDA Toolkit 13.0 is installed on target machine
- Add to system PATH if needed

### "Nsight Systems not found"
- Install Nsight Systems 2025.3.2 (or update hardcoded path in code)
- Or bundle nsys.exe with distribution

### "Missing DLL" errors
- Run from command prompt to see which DLL
- May need Visual C++ Redistributable on target machine

---

## Advanced: Bundle Nsight Systems

To make fully portable, bundle `nsys.exe`:

```python
# In cuda_profiler_gui.py, detect bundled nsys.exe
if getattr(sys, 'frozen', False):
    # Running as compiled exe
    base_path = sys._MEIPASS
    self.nsys_path = os.path.join(base_path, "nsys", "nsys.exe")
else:
    # Running as script
    self.nsys_path = r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.3.2\target-windows-x64\nsys.exe"
```

Then rebuild with nsys.exe included.

---

## Recommended Distribution

**For most users:** Use **Option 1** (single exe)
- Easiest to share
- No installation needed
- Just send the .exe file

**For power users:** Use **Option 2** (folder)
- Better performance
- Can share via zip file
