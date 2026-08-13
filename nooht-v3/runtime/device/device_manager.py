import os, platform, threading, ctypes
from typing import Optional, List
from dataclasses import dataclass
import logging
logger = logging.getLogger(__name__)

@dataclass
class DeviceHandle:
    device_id: int; device_type: str; name: str; total_memory_mb: int; available_memory_mb: int

class DeviceManager:
    def __init__(self):
        self._devices = []; self._allocated = {}; self._lock = threading.Lock(); self._discover()
        
    def _discover(self):
        self._devices.append(DeviceHandle(-1, "cpu", f"{platform.processor()} ({os.cpu_count()} cores)", 0, 0))
        cuda_count = self._detect_cuda_count()
        for i in range(cuda_count):
            name = self._get_cuda_name(i)
            vram_mb = self._get_cuda_vram_mb()
            self._devices.append(DeviceHandle(i, "cuda", name, vram_mb, vram_mb))
            self._allocated[i] = 0
        logger.info(f"Discovered {len(self._devices)} device(s).")

    def _load_cuda_lib(self):
        for path in ["libcudart.so", "libcudart.so.12", "libcudart.so.11", "cudart64_12.dll", "cudart64_11.dll"]:
            try: return ctypes.CDLL(path)
            except OSError: continue
        raise OSError("Cannot load CUDA runtime library")

    def _detect_cuda_count(self) -> int:
        try:
            lib = self._load_cuda_lib(); count = ctypes.c_int()
            return count.value if lib.cudaGetDeviceCount(ctypes.byref(count)) == 0 else 0
        except Exception: return 0

    def _get_cuda_name(self, device_id: int) -> str:
        try:
            lib = self._load_cuda_lib(); name = (ctypes.c_char * 256)()
            lib.cudaDeviceGetName(name, 256, device_id)
            return name.value.decode('utf-8', errors='replace')
        except Exception: return f"CUDA Device {device_id}"

    def _get_cuda_vram_mb(self) -> int:
        try:
            lib = self._load_cuda_lib(); free = ctypes.c_size_t(); total = ctypes.c_size_t()
            return total.value // (1024 * 1024) if lib.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)) == 0 else 0
        except Exception: return 0

    def allocate(self, device_id: int, memory_mb: int) -> bool:
        with self._lock:
            if device_id not in self._allocated: return False
            device = next((d for d in self._devices if d.device_id == device_id), None)
            if not device: return False
            available = device.total_memory_mb - self._allocated[device_id]
            if memory_mb > available: return False
            self._allocated[device_id] += memory_mb; return True

    def free(self, device_id: int, memory_mb: int):
        with self._lock:
            if device_id in self._allocated: self._allocated[device_id] = max(0, self._allocated[device_id] - memory_mb)

    def get_devices(self) -> List[DeviceHandle]: return self._devices
    def get_best_device(self, device_type: str = "auto") -> Optional[DeviceHandle]:
        if device_type == "auto":
            cuda_devices = [d for d in self._devices if d.device_type == "cuda"]
            if cuda_devices: return max(cuda_devices, key=lambda d: d.total_memory_mb)
            return self._devices[0]
        return next((d for d in self._devices if d.device_type == device_type), None)
