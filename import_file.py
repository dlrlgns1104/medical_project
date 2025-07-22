import torch
import random
import numpy as np

def set_seed(seed=10):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# seed 고정
set_seed(22)

print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device Count:", torch.cuda.device_count())
print("CUDA Current Device:", torch.cuda.current_device() if torch.cuda.is_available() else "No CUDA device")
