from safetensors.torch import save_file
import torch, sys


def convert_model(path:str) -> None:
    model = torch.load(path + ".bin", map_location="cpu")
    save_file(model, path + ".safetensors")
    

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} /outputs/pytorch_model.bin")
        sys.exit(1)
    convert_model(sys.argv[1])
    