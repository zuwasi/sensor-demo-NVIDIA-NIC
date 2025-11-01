import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import subprocess
import threading
import os
import json
import re
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class CUDAProfilerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CUDA Profiler - Nsight Systems")
        self.root.geometry("900x700")
        
        # Nsight Systems path
        self.nsys_path = r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.3.2\target-windows-x64\nsys.exe"
        self.nsys_ui_path = r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.3.2\target-windows-x64\nsys-ui.exe"
        
        # Variables
        self.exe_path = tk.StringVar()
        self.output_dir = tk.StringVar(value=os.path.expanduser("~\\Documents"))
        self.report_name = tk.StringVar(value="cuda_profile")
        self.last_report_path = None
        self.profiling = False
        self.metrics_file = os.path.join(self.output_dir.get(), "cuda_profiling_metrics.json")
        self.current_metrics = {}
        
        self.create_widgets()
    
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # Executable selection
        ttk.Label(main_frame, text="CUDA Executable:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.exe_path, width=60).grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(main_frame, text="Browse...", command=self.browse_exe).grid(row=0, column=2, padx=5, pady=5)
        
        # Output directory
        ttk.Label(main_frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_dir, width=60).grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(main_frame, text="Browse...", command=self.browse_output_dir).grid(row=1, column=2, padx=5, pady=5)
        
        # Report name
        ttk.Label(main_frame, text="Report Name:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.report_name, width=40).grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        self.profile_btn = ttk.Button(button_frame, text="▶ Profile Application", command=self.start_profiling)
        self.profile_btn.grid(row=0, column=0, padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="⬛ Stop", command=self.stop_profiling, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=1, padx=5)
        
        self.open_report_btn = ttk.Button(button_frame, text="📊 Open Last Report", command=self.open_report, state=tk.DISABLED)
        self.open_report_btn.grid(row=0, column=2, padx=5)
        
        ttk.Button(button_frame, text="📈 Compare Reports", command=self.compare_reports).grid(row=0, column=3, padx=5)
        
        ttk.Button(button_frame, text="📁 Open Output Folder", command=self.open_output_folder).grid(row=0, column=4, padx=5)
        
        # Output text area
        ttk.Label(main_frame, text="Profiling Output:").grid(row=4, column=0, sticky=(tk.W, tk.N), pady=5)
        
        self.output_text = scrolledtext.ScrolledText(main_frame, height=25, width=100, wrap=tk.WORD)
        self.output_text.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Configure text tags for colored output
        self.output_text.tag_config("error", foreground="red")
        self.output_text.tag_config("warning", foreground="orange")
        self.output_text.tag_config("success", foreground="green")
        self.output_text.tag_config("info", foreground="blue")
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
    
    def browse_exe(self):
        filename = filedialog.askopenfilename(
            title="Select CUDA Executable",
            filetypes=[("Executable files", "*.exe"), ("All files", "*.*")]
        )
        if filename:
            self.exe_path.set(filename)
    
    def browse_output_dir(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir.set(directory)
    
    def open_output_folder(self):
        output_path = self.output_dir.get()
        if os.path.exists(output_path):
            os.startfile(output_path)
        else:
            messagebox.showerror("Error", f"Directory does not exist: {output_path}")
    
    def open_report(self):
        if self.last_report_path and os.path.exists(self.last_report_path):
            try:
                subprocess.Popen([self.nsys_ui_path, self.last_report_path])
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open report: {e}")
        else:
            messagebox.showwarning("Warning", "No report available or report file not found")
    
    def append_output(self, text, tag=None):
        self.output_text.insert(tk.END, text, tag)
        self.output_text.see(tk.END)
        self.output_text.update()
    
    def start_profiling(self):
        exe = self.exe_path.get()
        if not exe or not os.path.exists(exe):
            messagebox.showerror("Error", "Please select a valid CUDA executable")
            return
        
        # Clear previous output
        self.output_text.delete(1.0, tk.END)
        
        # Generate timestamped report name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_base = f"{self.report_name.get()}_{timestamp}"
        output_path = os.path.join(self.output_dir.get(), report_base)
        self.last_report_path = f"{output_path}.nsys-rep"
        
        # Update UI
        self.profiling = True
        self.profile_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set("Profiling...")
        
        # Log start
        self.append_output(f"=== CUDA Profiling Started ===\n", "info")
        self.append_output(f"Executable: {exe}\n")
        self.append_output(f"Output: {output_path}\n")
        self.append_output(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n", "info")
        
        # Run profiling in thread
        thread = threading.Thread(target=self.run_profiling, args=(exe, output_path))
        thread.daemon = True
        thread.start()
    
    def run_profiling(self, exe, output_path):
        cmd = [
            self.nsys_path,
            "profile",
            "--stats=true",
            "--trace=cuda",  # Only trace CUDA (removed nvtx since we don't use it)
            "--sample=none",  # Disable CPU sampling (prevents warning)
            "--cpuctxsw=none",  # Disable CPU context switch tracing (prevents warning)
            "-o", output_path,
            exe
        ]
        
        self.current_metrics = {
            'timestamp': datetime.now().isoformat(),
            'executable': exe,
            'report_path': f"{output_path}.nsys-rep",
            'cuda_api_calls': {},
            'kernel_times': {},
            'memory_operations': {},
            'total_time': 0
        }
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            output_lines = []
            # Read output line by line
            for line in iter(process.stdout.readline, ''):
                if not self.profiling:
                    process.terminate()
                    break
                
                output_lines.append(line)
                
                # Colorize output
                tag = None
                if "ERROR" in line or "FAILED" in line:
                    tag = "error"
                elif "WARNING" in line or "SKIPPED" in line:
                    tag = "warning"
                elif "Generated:" in line or "Time (%)" in line:
                    tag = "success"
                
                self.append_output(line, tag)
            
            process.wait()
            
            # Parse metrics from output
            self.parse_metrics(output_lines)
            
            # Update UI on completion
            self.root.after(0, self.profiling_complete, process.returncode)
            
        except Exception as e:
            self.root.after(0, self.profiling_error, str(e))
    
    def parse_metrics(self, output_lines):
        """Parse profiling metrics from nsys output"""
        try:
            in_cuda_api_section = False
            in_kernel_section = False
            in_memory_section = False
            
            for line in output_lines:
                # CUDA API calls section
                if "Executing 'cuda_api_sum' stats report" in line:
                    in_cuda_api_section = True
                    in_kernel_section = False
                    in_memory_section = False
                    continue
                elif "Executing 'cuda_gpu_kern_sum' stats report" in line:
                    in_kernel_section = True
                    in_cuda_api_section = False
                    in_memory_section = False
                    continue
                elif "Executing 'cuda_gpu_mem_time_sum' stats report" in line:
                    in_memory_section = True
                    in_cuda_api_section = False
                    in_kernel_section = False
                    continue
                elif "Executing" in line and "stats report" in line:
                    in_cuda_api_section = False
                    in_kernel_section = False
                    in_memory_section = False
                    continue
                
                # Parse CUDA API calls
                if in_cuda_api_section:
                    match = re.search(r'^\s+([\d.]+)\s+([\d.]+)\s+\d+\s+[\d.]+\s+[\d.]+\s+\d+\s+\d+\s+[\d.]+\s+(.+)$', line)
                    if match:
                        time_ns = float(match.group(2))
                        name = match.group(3).strip()
                        self.current_metrics['cuda_api_calls'][name] = time_ns
                        self.current_metrics['total_time'] += time_ns
                
                # Parse kernel times
                if in_kernel_section:
                    match = re.search(r'^\s+[\d.]+\s+([\d.]+)\s+\d+\s+[\d.]+\s+[\d.]+\s+\d+\s+\d+\s+[\d.]+\s+(.+)$', line)
                    if match:
                        time_ns = float(match.group(1))
                        name = match.group(2).strip()
                        self.current_metrics['kernel_times'][name] = time_ns
                
                # Parse memory operations
                if in_memory_section:
                    match = re.search(r'^\s+[\d.]+\s+([\d.]+)\s+\d+\s+[\d.]+\s+[\d.]+\s+\d+\s+\d+\s+[\d.]+\s+(.+)$', line)
                    if match:
                        time_ns = float(match.group(1))
                        name = match.group(2).strip()
                        self.current_metrics['memory_operations'][name] = time_ns
        
        except Exception as e:
            print(f"Error parsing metrics: {e}")
    
    def save_metrics(self):
        """Save metrics to JSON file"""
        try:
            # Load existing metrics
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r') as f:
                    all_metrics = json.load(f)
            else:
                all_metrics = []
            
            # Append new metrics
            all_metrics.append(self.current_metrics)
            
            # Save back to file
            with open(self.metrics_file, 'w') as f:
                json.dump(all_metrics, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving metrics: {e}")
            return False
    
    def profiling_complete(self, return_code):
        self.profiling = False
        self.profile_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        
        if return_code == 0:
            # Save metrics
            if self.save_metrics():
                self.append_output(f"\n✓ Metrics saved to {self.metrics_file}\n", "success")
            
            self.append_output(f"\n=== Profiling Completed Successfully ===\n", "success")
            self.status_var.set("Profiling completed successfully")
            self.open_report_btn.config(state=tk.NORMAL)
            
            # Ask to open report
            if messagebox.askyesno("Success", "Profiling completed! Open report in Nsight Systems?"):
                self.open_report()
        else:
            self.append_output(f"\n=== Profiling Failed (Exit code: {return_code}) ===\n", "error")
            self.status_var.set("Profiling failed")
    
    def profiling_error(self, error_msg):
        self.profiling = False
        self.profile_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.append_output(f"\nERROR: {error_msg}\n", "error")
        self.status_var.set("Error during profiling")
        messagebox.showerror("Profiling Error", error_msg)
    
    def stop_profiling(self):
        self.profiling = False
        self.append_output("\n=== Profiling Stopped by User ===\n", "warning")
        self.status_var.set("Profiling stopped")
    
    def compare_reports(self):
        """Open comparison window for multiple reports"""
        if not os.path.exists(self.metrics_file):
            messagebox.showwarning("No Data", "No profiling metrics available. Run some profiles first.")
            return
        
        try:
            with open(self.metrics_file, 'r') as f:
                all_metrics = json.load(f)
            
            if len(all_metrics) < 2:
                messagebox.showwarning("Insufficient Data", "Need at least 2 profiling runs to compare.")
                return
            
            # Create comparison window
            ComparisonWindow(self.root, all_metrics)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load metrics: {e}")


class ComparisonWindow:
    def __init__(self, parent, metrics_data):
        self.window = tk.Toplevel(parent)
        self.window.title("CUDA Profiling Comparison")
        self.window.geometry("1200x800")
        
        self.metrics_data = metrics_data
        self.selected_indices = []
        self.chart_windows = []  # Track chart windows
        
        # Ensure proper cleanup when window closes
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.create_widgets()
        self.populate_list()
    
    def on_close(self):
        """Clean up chart windows and close"""
        for chart_win in self.chart_windows:
            try:
                chart_win.destroy()
            except:
                pass
        self.window.destroy()
    
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Instructions
        ttk.Label(main_frame, text="Select reports to compare (Ctrl+Click for multiple):").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        # Listbox with scrollbar
        list_frame = ttk.Frame(main_frame)
        list_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, height=10, yscrollcommand=scrollbar.set)
        self.listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.config(command=self.listbox.yview)
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=2, column=0, pady=10)
        
        ttk.Button(btn_frame, text="📊 Compare Kernel Times", command=lambda: self.show_comparison("kernel")).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="💾 Compare Memory Ops", command=lambda: self.show_comparison("memory")).grid(row=0, column=1, padx=5)
        ttk.Button(btn_frame, text="⚡ Compare CUDA API", command=lambda: self.show_comparison("api")).grid(row=0, column=2, padx=5)
        ttk.Button(btn_frame, text="📈 Compare Total Time", command=lambda: self.show_comparison("total")).grid(row=0, column=3, padx=5)
    
    def populate_list(self):
        """Populate listbox with available reports"""
        for i, metric in enumerate(self.metrics_data):
            timestamp = datetime.fromisoformat(metric['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
            exe_name = os.path.basename(metric['executable'])
            total_time_ms = metric.get('total_time', 0) / 1_000_000  # Convert ns to ms
            
            display_text = f"{i+1}. {timestamp} | {exe_name} | Total: {total_time_ms:.2f}ms"
            self.listbox.insert(tk.END, display_text)
    
    def show_comparison(self, comparison_type):
        """Show comparison chart based on selected reports"""
        selected = self.listbox.curselection()
        
        if len(selected) < 2:
            messagebox.showwarning("Selection Required", "Please select at least 2 reports to compare")
            return
        
        selected_metrics = [self.metrics_data[i] for i in selected]
        
        # Create chart window
        chart_window = tk.Toplevel(self.window)
        chart_window.title(f"Comparison - {comparison_type.title()}")
        chart_window.geometry("1000x600")
        
        # Track this window for cleanup
        self.chart_windows.append(chart_window)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if comparison_type == "total":
            self.plot_total_time(ax, selected_metrics)
        elif comparison_type == "kernel":
            self.plot_kernel_times(ax, selected_metrics)
        elif comparison_type == "memory":
            self.plot_memory_ops(ax, selected_metrics)
        elif comparison_type == "api":
            self.plot_api_calls(ax, selected_metrics)
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=chart_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(canvas, chart_window)
        toolbar.update()
        
        # Clean up matplotlib figure when window closes
        def on_chart_close():
            plt.close(fig)
            if chart_window in self.chart_windows:
                self.chart_windows.remove(chart_window)
            chart_window.destroy()
        
        chart_window.protocol("WM_DELETE_WINDOW", on_chart_close)
    
    def plot_total_time(self, ax, selected_metrics):
        """Plot total execution time with trend line"""
        timestamps = [datetime.fromisoformat(m['timestamp']) for m in selected_metrics]
        total_times = [m.get('total_time', 0) / 1_000_000 for m in selected_metrics]  # Convert to ms
        
        # Get report names from report_path
        report_labels = []
        for m in selected_metrics:
            report_path = m.get('report_path', '')
            if report_path:
                # Extract filename without extension
                report_name = os.path.splitext(os.path.basename(report_path))[0]
                report_labels.append(report_name)
            else:
                report_labels.append(f"Run {len(report_labels)+1}")
        
        # Create bar chart
        x_pos = np.arange(len(timestamps))
        bars = ax.bar(x_pos, total_times, alpha=0.7, color='steelblue', edgecolor='black')
        
        # Add value labels on bars
        for i, (bar, time) in enumerate(zip(bars, total_times)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time:.2f}ms',
                   ha='center', va='bottom', fontsize=9)
        
        # Add trend line
        if len(x_pos) >= 2:
            z = np.polyfit(x_pos, total_times, 1)
            p = np.poly1d(z)
            ax.plot(x_pos, p(x_pos), "r--", linewidth=2, label=f'Trend: {z[0]:.2f}ms/run')
            
            # Add improvement percentage
            if total_times[0] != 0:
                improvement = ((total_times[0] - total_times[-1]) / total_times[0]) * 100
                trend_text = f"Change: {improvement:+.1f}%"
                ax.text(0.02, 0.98, trend_text, transform=ax.transAxes,
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Format x-axis
        ax.set_xlabel('Report Name', fontsize=12)
        ax.set_ylabel('Total Time (ms)', fontsize=12)
        ax.set_title('Total Execution Time Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(report_labels, rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
    
    def plot_kernel_times(self, ax, selected_metrics):
        """Plot kernel execution times"""
        # Collect all unique kernel names
        all_kernels = set()
        for m in selected_metrics:
            all_kernels.update(m.get('kernel_times', {}).keys())
        
        if not all_kernels:
            ax.text(0.5, 0.5, 'No kernel data available', ha='center', va='center', fontsize=14)
            return
        
        # Get report names
        report_labels = []
        for m in selected_metrics:
            report_path = m.get('report_path', '')
            if report_path:
                report_name = os.path.splitext(os.path.basename(report_path))[0]
                report_labels.append(report_name)
            else:
                report_labels.append(f"Run {len(report_labels)+1}")
        
        # Prepare data
        timestamps = [datetime.fromisoformat(m['timestamp']) for m in selected_metrics]
        kernel_data = {kernel: [] for kernel in all_kernels}
        
        for m in selected_metrics:
            for kernel in all_kernels:
                time_ns = m.get('kernel_times', {}).get(kernel, 0)
                kernel_data[kernel].append(time_ns / 1000)  # Convert to microseconds
        
        # Create grouped bar chart
        x_pos = np.arange(len(timestamps))
        width = 0.8 / len(all_kernels) if all_kernels else 0.8
        
        for i, (kernel, times) in enumerate(kernel_data.items()):
            offset = (i - len(all_kernels)/2) * width + width/2
            bars = ax.bar(x_pos + offset, times, width, label=kernel, alpha=0.8)
            
            # Add trend line for each kernel
            if len(x_pos) >= 2 and sum(times) > 0:
                z = np.polyfit(x_pos, times, 1)
                p = np.poly1d(z)
                ax.plot(x_pos + offset, p(x_pos), "--", linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('Report Name', fontsize=12)
        ax.set_ylabel('Kernel Time (μs)', fontsize=12)
        ax.set_title('Kernel Execution Time Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(report_labels, rotation=45, ha='right', fontsize=8)
        ax.legend(loc='best')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
    
    def plot_memory_ops(self, ax, selected_metrics):
        """Plot memory operation times"""
        all_ops = set()
        for m in selected_metrics:
            all_ops.update(m.get('memory_operations', {}).keys())
        
        if not all_ops:
            ax.text(0.5, 0.5, 'No memory operation data available', ha='center', va='center', fontsize=14)
            return
        
        # Get report names
        report_labels = []
        for m in selected_metrics:
            report_path = m.get('report_path', '')
            if report_path:
                report_name = os.path.splitext(os.path.basename(report_path))[0]
                report_labels.append(report_name)
            else:
                report_labels.append(f"Run {len(report_labels)+1}")
        
        timestamps = [datetime.fromisoformat(m['timestamp']) for m in selected_metrics]
        mem_data = {op: [] for op in all_ops}
        
        for m in selected_metrics:
            for op in all_ops:
                time_ns = m.get('memory_operations', {}).get(op, 0)
                mem_data[op].append(time_ns / 1000)  # Convert to microseconds
        
        x_pos = np.arange(len(timestamps))
        width = 0.8 / len(all_ops) if all_ops else 0.8
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(all_ops)))
        
        for i, (op, times) in enumerate(mem_data.items()):
            offset = (i - len(all_ops)/2) * width + width/2
            bars = ax.bar(x_pos + offset, times, width, label=op, alpha=0.8, color=colors[i])
            
            if len(x_pos) >= 2 and sum(times) > 0:
                z = np.polyfit(x_pos, times, 1)
                p = np.poly1d(z)
                ax.plot(x_pos + offset, p(x_pos), "--", linewidth=1.5, alpha=0.7, color=colors[i])
        
        ax.set_xlabel('Report Name', fontsize=12)
        ax.set_ylabel('Time (μs)', fontsize=12)
        ax.set_title('Memory Operations Time Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(report_labels, rotation=45, ha='right', fontsize=8)
        ax.legend(loc='best', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
    
    def plot_api_calls(self, ax, selected_metrics):
        """Plot top CUDA API call times"""
        # Get top 5 most time-consuming API calls from all runs
        all_api_times = {}
        for m in selected_metrics:
            for api, time in m.get('cuda_api_calls', {}).items():
                all_api_times[api] = all_api_times.get(api, 0) + time
        
        if not all_api_times:
            ax.text(0.5, 0.5, 'No CUDA API data available', ha='center', va='center', fontsize=14)
            return
        
        # Get top 5 APIs
        top_apis = sorted(all_api_times.items(), key=lambda x: x[1], reverse=True)[:5]
        top_api_names = [api for api, _ in top_apis]
        
        # Get report names
        report_labels = []
        for m in selected_metrics:
            report_path = m.get('report_path', '')
            if report_path:
                report_name = os.path.splitext(os.path.basename(report_path))[0]
                report_labels.append(report_name)
            else:
                report_labels.append(f"Run {len(report_labels)+1}")
        
        timestamps = [datetime.fromisoformat(m['timestamp']) for m in selected_metrics]
        api_data = {api: [] for api in top_api_names}
        
        for m in selected_metrics:
            for api in top_api_names:
                time_ns = m.get('cuda_api_calls', {}).get(api, 0)
                api_data[api].append(time_ns / 1_000_000)  # Convert to ms
        
        x_pos = np.arange(len(timestamps))
        width = 0.8 / len(top_api_names) if top_api_names else 0.8
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_api_names)))
        
        for i, (api, times) in enumerate(api_data.items()):
            offset = (i - len(top_api_names)/2) * width + width/2
            bars = ax.bar(x_pos + offset, times, width, label=api, alpha=0.8, color=colors[i])
            
            if len(x_pos) >= 2 and sum(times) > 0:
                z = np.polyfit(x_pos, times, 1)
                p = np.poly1d(z)
                ax.plot(x_pos + offset, p(x_pos), "--", linewidth=1.5, alpha=0.7, color=colors[i])
        
        ax.set_xlabel('Report Name', fontsize=12)
        ax.set_ylabel('Time (ms)', fontsize=12)
        ax.set_title('Top 5 CUDA API Calls Time Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(report_labels, rotation=45, ha='right', fontsize=8)
        ax.legend(loc='best', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()


if __name__ == "__main__":
    root = tk.Tk()
    app = CUDAProfilerGUI(root)
    
    # Ensure clean exit
    def on_closing():
        plt.close('all')  # Close all matplotlib figures
        root.quit()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        on_closing()
    finally:
        plt.close('all')
