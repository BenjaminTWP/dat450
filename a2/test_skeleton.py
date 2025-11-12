from A2_skeleton_provided import A2ModelConfig, A2MLP, A2Attention, A2RotaryEmbedding
import torch
from transformers import PretrainedConfig

config = A2ModelConfig(
    vocab_size=150000,
    hidden_size=10,
    max_position_embeddings=100000,
    rms_norm_eps=0.001,
    num_attention_heads=2,
    rope_theta=2,
    hidden_act='silu',
    intermediate_size=64,
    num_hidden_layers=32,
    embedding_dims=128
)

tensor = torch.ones((5, 20, 10)) * 100

#mlp_model = A2MLP(config)
#out = mlp_model(tensor)

rotary_model = A2RotaryEmbedding(config)
rotation_emb = rotary_model(tensor)

attention_model = A2Attention(config)
out = attention_model(tensor, rotation_emb)

print(out.shape)