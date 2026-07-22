from pathlib import Path
import sys

import gradio
import torch
import transformers

base = Path(__file__).resolve().parent
preferred = base / "fine_tuned_transformer"

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("Gradio:", gradio.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

if preferred.is_dir():
    required = [
        preferred / "config.json",
        preferred / "model.safetensors",
    ]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        print("Model folder exists, but files are missing:", missing)
    else:
        print("Model folder looks ready:", preferred)
else:
    print(
        "Model folder is not yet at the preferred location:",
        preferred,
    )
    print(
        "The app also performs an automatic recursive search when it starts."
    )
