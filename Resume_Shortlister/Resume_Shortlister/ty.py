import torch

print(f"GPU Available: {torch.cuda.is_available()}")  # Should return True
print(f"GPU Name: {torch.cuda.get_device_name(0)}")    # e.g., "NVIDIA RTX 3060"
print(f"CUDA Version: {torch.version.cuda}")           # e.g., "11.7"