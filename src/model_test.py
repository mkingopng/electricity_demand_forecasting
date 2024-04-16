"""
chek to ensure that model checkpoint integrity is OK
"""
import torch
try:
    state_dict = torch.load('model_checkpoint.pt')
    print("State dict keys:", state_dict.keys())
except Exception as e:
    print("Failed to load the state dict:", e)
