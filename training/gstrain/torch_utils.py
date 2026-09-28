import torch


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device(device: torch.device | None = None) -> str:
    device = device or choose_device()

    if device.type != "cuda":
        return f"{device} (CPU fallback)"

    index = device.index or 0
    name = torch.cuda.get_device_name(index)
    total_gb = torch.cuda.get_device_properties(index).total_memory / 1024**3

    return f"{device} - {name} ({total_gb:.1f} GB)"
