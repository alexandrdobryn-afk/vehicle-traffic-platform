import logging
from typing import List
from app.schemas.schemas import GPUInfo

logger = logging.getLogger(__name__)


class GPUService:

    def get_gpu_info(self) -> List[GPUInfo]:
        """Return info for all available GPUs."""
        gpus = []

        try:
            import GPUtil
            for g in GPUtil.getGPUs():
                gpus.append(GPUInfo(
                    id=g.id,
                    name=g.name,
                    memory_total_mb=g.memoryTotal,
                    memory_used_mb=g.memoryUsed,
                    memory_free_mb=g.memoryFree,
                    utilization_pct=g.load * 100,
                    temperature=g.temperature,
                    is_available=g.load < 0.9 and g.memoryFree > 512,
                ))
        except Exception:
            pass

        if not gpus:
            # Try via torch
            try:
                import torch
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        props = torch.cuda.get_device_properties(i)
                        total = props.total_memory / 1e6
                        used = (props.total_memory - torch.cuda.mem_get_info(i)[0]) / 1e6
                        free = torch.cuda.mem_get_info(i)[0] / 1e6
                        gpus.append(GPUInfo(
                            id=i,
                            name=props.name,
                            memory_total_mb=total,
                            memory_used_mb=used,
                            memory_free_mb=free,
                            utilization_pct=0,
                            temperature=None,
                            is_available=free > 512,
                        ))
            except Exception:
                pass

        return gpus

    def get_system_stats(self) -> dict:
        import psutil
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage(str(__import__("pathlib").Path(__file__).parent))
        return {
            "cpu_pct": cpu,
            "ram_total_gb": round(mem.total / 1e9, 1),
            "ram_used_gb": round(mem.used / 1e9, 1),
            "ram_pct": mem.percent,
            "disk_total_gb": round(disk.total / 1e9, 1),
            "disk_used_gb": round(disk.used / 1e9, 1),
            "disk_pct": disk.percent,
        }

    def is_gpu_available(self) -> bool:
        gpus = self.get_gpu_info()
        return any(g.is_available for g in gpus)


gpu_service = GPUService()
