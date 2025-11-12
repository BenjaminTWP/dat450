from A2_skeleton_provided import A2ModelConfig, A2MLP, A2Attention
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