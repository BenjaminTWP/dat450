from A2_skeleton import *
from torch import nn
import numpy as np


def sanity_mlp_rms():
    config = A2ModelConfig(hidden_size=64, intermediate_size=45,
                             rms_norm_eps=1e-2, rope_theta=10000.0,
                             num_attention_heads=4,vocab_size=100)
    a2mlp = A2MLP(config)

    random_vector = torch.tensor(np.random.rand(10, 20, config.hidden_size), dtype=torch.float32)
    print(f"\n#\nSanity check of vector (shape={random_vector.shape}) for MLP layer\n#")

    out = a2mlp(random_vector)
    print(f"Shape after passing through MLP is {out.shape}\n#")

    normalizer = nn.RMSNorm(normalized_shape=config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True)
    out_norm = normalizer(out)
    print(f"Shape after passing through normalizer is {out_norm.shape}")

    attention = A2Attention(config)
    rotary_emb = A2RotaryEmbedding(config)
    rotary_emb_result = rotary_emb(out_norm)
    out_attention = attention(out_norm, rotary_emb_result, sanity=True)
    print(f"#\nShape after MHA: {out_attention.shape}")

    decoder = A2DecoderLayer(config)
    out_decoder = decoder(out_norm, rotary_emb_result)
    print(f"#\nShape after decoder layer: {out_decoder.shape}")



if __name__ == "__main__":
    print("#", flush=True)
    sanity_mlp_rms()
