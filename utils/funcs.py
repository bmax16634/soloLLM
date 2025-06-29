import torch, os
from safetensors.torch import load_file

from models.soloGPT_v1_model import SoloGPT_v1









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




def load_model(config, checkpoint_path=None, device=None):
    model = SoloGPT_v1(config).to(device)

    if checkpoint_path is not None:
        ext = os.path.splitext(checkpoint_path)[1].lower()

        if ext in (".bin", ".pth"):
            state_dict = torch.load(checkpoint_path, map_location=device)
        elif ext == ".safetensors":
            device_str = f"cuda:{device.index or 0}" if device.type == "cuda" else "cpu"
            state_dict = load_file(checkpoint_path, device=device_str)
        else:
            raise ValueError(f"Unsupported checkpoint format: '{ext}'")

        model.load_state_dict(state_dict)

    return model