import time
import os
import sys

try:
    import psutil
except ImportError:
    psutil = None

class PerformanceBenchmarker:
    def __init__(self):
        self.process = psutil.Process(os.getpid()) if psutil else None
        
    def start_timing(self) -> float:
        return time.perf_counter()
        
    def end_timing(self, start_time: float) -> int:
        return int((time.perf_counter() - start_time) * 1000)
        
    def get_ram_usage_mb(self) -> float:
        """Returns current process RAM memory usage in MB."""
        if self.process:
            try:
                return float(self.process.memory_info().rss / (1024 * 1024))
            except Exception:
                pass
        # Windows-specific fallback using ctypes
        if sys.platform == "win32":
            try:
                import ctypes
                from ctypes import wintypes
                
                class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                    _fields_ = [
                        ("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t),
                    ]
                
                GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo
                GetCurrentProcess = ctypes.windll.kernel32.GetCurrentProcess
                
                process_handle = GetCurrentProcess()
                counters = PROCESS_MEMORY_COUNTERS()
                counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
                
                if GetProcessMemoryInfo(process_handle, ctypes.byref(counters), counters.cb):
                    return float(counters.WorkingSetSize / (1024 * 1024))
            except Exception:
                pass
        return 0.0
        
    def get_gpu_vram_usage(self) -> float:
        """Returns GPU VRAM memory usage in MB if PyTorch is active, else returns 0.0."""
        try:
            import torch
            if torch.cuda.is_available():
                return float(torch.cuda.memory_allocated() / (1024 * 1024))
        except ImportError:
            pass
        return 0.0

    def get_gpu_utilization(self) -> float:
        """Queries nvidia-smi for GPU utilization, falls back to 0.0 if CPU-only."""
        try:
            import subprocess
            # Run nvidia-smi command to get GPU utilization percentage
            res = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL
            )
            return float(res.decode("utf-8").strip())
        except Exception:
            pass
        return 0.0
