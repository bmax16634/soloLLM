import torch










def get_device() -> tuple[torch.device, str]:
    if torch.cuda.is_available():
        # default to cuda:0; if you wanted another GPU, you'd set the index in torch.device(...)
        device = torch.device("cuda")
        # device.index is None when you do torch.device("cuda"), so default to 0
        device_str = f"cuda:{device.index if device.index is not None else 0}"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_str = "mps"
    else:
        device = torch.device("cpu")
        device_str = "cpu"

    return device, device_str






