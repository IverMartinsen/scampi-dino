"""
To easily get the number of trainable parameters.
"""
import numpy as np
from vision_transformer import vit_tiny, vit_small

vt = vit_tiny()
vs = vit_small()

trainable_params_vt = sum(p.numel() for p in vt.parameters() if p.requires_grad) / 1e6
trainable_params_vs = sum(p.numel() for p in vs.parameters() if p.requires_grad) / 1e6

print(f"Trainable parameters: {np.round(trainable_params_vt, 2)}M for ViT-Tiny and {np.round(trainable_params_vs, 2)}M for ViT-Small.")