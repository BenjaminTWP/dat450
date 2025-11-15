import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig


class A2ModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the Transformer language model."""
    def __init__(self, vocab_size=None, hidden_size=None, intermediate_size=None, num_attention_heads=None, 
                 num_hidden_layers=None, rope_theta=None, hidden_act='silu', max_position_embeddings=None,
                  embedding_dims=None, rms_norm_eps=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.num_attention_heads = num_attention_heads
        self.rope_theta = rope_theta
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers



class A2MLP(nn.Module):
    """The MLP layer of the Transformer. Uses the SwiGLU architecture."""
    def __init__(self, config):
        super().__init__()
        assert(config.hidden_act == 'silu')
        self.linear_1 = nn.Linear(in_features=config.hidden_size, out_features=config.intermediate_size, bias=False)
        self.linear_2 = nn.Linear(in_features=config.hidden_size, out_features=config.intermediate_size, bias=False)
        self.linear_3 = nn.Linear(in_features=config.intermediate_size, out_features=config.hidden_size, bias=False)
        self.SiLU = nn.SiLU()
        

    def forward(self, hidden_states):
        out_1 = self.linear_1(hidden_states)
        out_2 = self.linear_2(hidden_states)
        out_1 = self.SiLU(out_1)
        out = torch.mul(out_1, out_2)
        return self.linear_3(out) 

# This is optional, since you can use PyTorch's RMSNorm.
class A2RMSNorm(nn.Module):
    """RMS layer normalization."""
    def __init__(self, config):
        super().__init__()
        # TODO: Use config.rms_norm_eps
        # TODO: initalize weights here

    def forward(self, hidden_states):
        ...


class A2Attention(nn.Module):
    """The multi-head attention layer of the Transformer. Uses standard scaled dot-product attention with causal masking."""
    
    def __init__(self, config):
        super().__init__()
        self.W_Q = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)
        self.W_K = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)
        self.W_V = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)
        self.W_O = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)
        self.key_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True)
        self.query_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True)
        self.config = config

        self.total_dims = config.hidden_size
        self.nr_heads = config.num_attention_heads

        if self.total_dims % self.nr_heads != 0:
            raise Exception("The number of dims must be equally divided acrosss the heads")

        self.dims_per_head = self.total_dims // self.nr_heads


        # TODO: set up W_q, W_k, W_v, W_o here
        # TODO: set up normalizers here

    def forward(self, hidden_states, rope_rotations):
        Q = self.query_norm(self.W_Q(hidden_states))
        K = self.key_norm(self.W_K(hidden_states))
        V = self.W_V(hidden_states) 

        batch_size, text_length, dims = Q.shape    

        Q = Q.view(batch_size, text_length, self.nr_heads,  self.dims_per_head).transpose(1, 2)
        K = K.view(batch_size, text_length, self.nr_heads,  self.dims_per_head).transpose(1, 2)
        V = V.view(batch_size, text_length, self.nr_heads,  self.dims_per_head).transpose(1, 2)

        Q_emb, K_emb = apply_rotary_pos_emb(Q, K, rope_rotations)
        attn_out = nn.functional.scaled_dot_product_attention(Q_emb, K_emb, V, is_causal=True)

        attn_out = attn_out.transpose(1, 2).reshape(batch_size, text_length, -1)

        return self.W_O(attn_out)


class A2DecoderLayer(nn.Module):
    """A complete Transformer decoder layer."""
    def __init__(self, config):
        super().__init__()
        self.MHA = A2Attention(config)
        self.MLP = A2MLP(config)
        self.rms_1 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True)
        self.rms_2 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True)

        # TODO: set up attention, MLP, and normalizers here.

    def forward(self, hidden_states, rope_rotations):
        out = self.MHA(hidden_states, rope_rotations)
        out = self.rms_1(out)

        tmp = torch.add(hidden_states, out)
        out = self.MLP(tmp)
        return torch.add(tmp, self.rms_2(out))


class A2Transformer(PreTrainedModel):
    """A language model based on the Transformer architecture."""
    
    config_class = A2ModelConfig

    def __init__(self, config):
        super().__init__(config)

        self.rotary_emb = A2RotaryEmbedding(config)
        self.embedding = torch.nn.Embedding(num_embeddings=config.vocab_size,
                                            embedding_dim=config.hidden_size)

        self.layers = nn.ModuleList([
            A2DecoderLayer(config) for i in range(config.num_hidden_layers)
        ])
        self.rms = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True)
        self.linear = nn.Linear(in_features=config.hidden_size, out_features=config.vocab_size, bias=False)

        # TODO: Set up the other components here.
        # TODO: put all transformer decoder layers in a ModuleList.

        # This line should be called after you have set up all components.
        self.post_init()


    def forward(self, input_ids):
        rope_rotations = self.rotary_emb(input_ids) # pass this to all the transformer decoder layers

        input = self.embedding(input_ids)

        for decoder in self.layers:
            input = decoder(input, rope_rotations)

        out = self.rms(input)
        return self.linear(out)

        # TODO: Call embedding, transformer decoder layers, last normalizer, and unembedding.
        ...


def apply_rotary_pos_emb(q, k, rope_rotations, unsqueeze_dim=1):
    """Applies precomputed RoPE rotations to the query and key representations."""
    assert(q.shape == k.shape)
    assert(len(q.shape) == 4)
    cos, sin = rope_rotations
    assert(q.shape[2] == cos.shape[1])
    assert(q.shape[3] == cos.shape[2])    
    q_type, k_type = q.dtype, k.dtype
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q_type), k_embed.to(k_type)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class A2RotaryEmbedding(nn.Module):
    """RoPE position representation for use in Transformer attention."""

    def __init__(self, config, device=None):
        super().__init__()
        rope_theta = config.rope_theta
        head_dim = config.hidden_size // config.num_attention_heads
        partial_rotary_factor = 1.0
        dim = int(head_dim * partial_rotary_factor)
        self.inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))

    @torch.no_grad()
    def forward(self, x):
        position_ids = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            return cos, sin

