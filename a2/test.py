from A2_skeleton import SwiGLUModelConfig, SwiGLU
import torch
from transformers import PretrainedConfig


config = SwiGLUModelConfig(
    hidden_size = 10,
    intermediate_size = 100,
    rms_norm_eps = 0.001,
)

model = SwiGLU(config)

tensor = torch.ones((5, 20, 10)) * 100
out = model(tensor)

print(out.shape)
#print(out)