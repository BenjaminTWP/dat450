

import torch
from torch import nn
from transformers import (
    PreTrainedModel, 
    PretrainedConfig,
)
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.generation import GenerationMixin

class ModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the Transformer language model."""
    def __init__(self, vocab_size=None, hidden_size=None, intermediate_size=None, num_attention_heads=None, 
                 num_hidden_layers=None,
                 rope_theta=None, hidden_act='silu', max_position_embeddings=None, rms_norm_eps=None, **kwargs):
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



class MLP(nn.Module):
    """The MLP layer of the Transformer. Uses the SwiGLU architecture."""
    def __init__(self, config):
        super().__init__()
        assert(config.hidden_act == 'silu')

        self.lin_r = nn.Linear(in_features=config.hidden_size, out_features=config.intermediate_size,  bias=False)
        self.lin_l = nn.Linear(in_features=config.hidden_size ,out_features=config.intermediate_size,  bias=False)
        self.act_silu = nn.SiLU()
        self.lin_last = nn.Linear(in_features=config.intermediate_size, out_features=config.hidden_size, bias=False)


    def forward(self, hidden_states):
        out_right = self.lin_r(hidden_states)
        out_left = self.act_silu(self.lin_l(hidden_states))
        product = out_left * out_right
        return self.lin_last(product)



class Attention(nn.Module):
    """The multi-head attention layer of the Transformer. Uses standard scaled dot-product attention with causal masking."""
    
    def __init__(self, config, masked):
        super().__init__()
        assert(config.hidden_size % config.num_attention_heads == 0)
        self.config = config
        self.masked = masked
        
        self.W_q = nn.Linear(config.hidden_size,config.hidden_size, bias=False)
        self.W_k = nn.Linear(config.hidden_size,config.hidden_size, bias=False)
        self.W_v = nn.Linear(config.hidden_size,config.hidden_size, bias=False)
        self.W_o = nn.Linear(config.hidden_size,config.hidden_size, bias=False)
        self.norm_queries = nn.RMSNorm(normalized_shape=config.hidden_size, 
                               eps=config.rms_norm_eps, elementwise_affine=True)
        self.norm_keys = nn.RMSNorm(normalized_shape=config.hidden_size, 
                               eps=config.rms_norm_eps, elementwise_affine=True)


    def forward(self, hidden_states_encoder, hidden_states_decoder=None, rope_rotations=None, sanity=False): # forwards cross attention
        
        if hidden_states_decoder is not None:
            queries = self.W_q(hidden_states_decoder)
        else:
            queries = self.W_q(hidden_states_encoder)

        keys = self.W_k(hidden_states_encoder)
        values = self.W_v(hidden_states_encoder)

        queries = self.norm_queries(queries)
        keys = self.norm_keys(keys)

        b_q, m_q = queries.shape[:2]
        b_k, m_k = keys.shape[:2]
        b_v, m_v = values.shape[:2]
        
        n_h = self.config.num_attention_heads
        d_h = self.config.hidden_size // n_h

        if sanity:
            print(f"#\nShape of value vector is {values.shape}")
            print(f"#\nShape of key vector is {keys.shape}")
            print(f"#\nShape of query vector is {queries.shape}")

        queries = queries.view(b_q, m_q, n_h, d_h).transpose(1, 2)
        keys = keys.view(b_k, m_k, n_h, d_h).transpose(1, 2)
        values = values.view(b_v, m_v, n_h, d_h).transpose(1, 2)

        if rope_rotations is not None:
            queries, keys = apply_rotary_pos_emb(queries, keys, rope_rotations)

        attn_out = nn.functional.scaled_dot_product_attention(queries, keys, values, is_causal=self.masked)
        attn_out= attn_out.transpose(1, 2).reshape(b_q, m_q, hidden_states_encoder.shape[2])

        return self.W_o(attn_out)


class EncoderLayer(nn.Module):
    """A complete Transformer encoder layer."""
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config, False) # unmasked self attention
        self.mlp = MLP(config)
        self.norm1 = nn.RMSNorm(normalized_shape=config.hidden_size, 
                               eps=config.rms_norm_eps, elementwise_affine=True)
        self.norm2 = nn.RMSNorm(normalized_shape=config.hidden_size, 
                               eps=config.rms_norm_eps, elementwise_affine=True)

    def forward(self, hidden_states, rope_rotations):
        attention_out_norm = self.norm1(self.attention(hidden_states, rope_rotations=rope_rotations))
        activation_norm = self.norm2(self.mlp(attention_out_norm + hidden_states))
        return attention_out_norm + activation_norm



class DecoderLayer(nn.Module):
    """A complete Transformer decoder layer."""
    def __init__(self, config):
        super().__init__()
        self.self_attention = Attention(config, True) # masked self attention
        self.cross_attention = Attention(config, False) # unmasked cross attention

        self.mlp = MLP(config)
        self.norm1 = nn.RMSNorm(normalized_shape=config.hidden_size, 
                               eps=config.rms_norm_eps, elementwise_affine=True)
        self.norm2 = nn.RMSNorm(normalized_shape=config.hidden_size, 
                               eps=config.rms_norm_eps, elementwise_affine=True)
        self.norm3 = nn.RMSNorm(normalized_shape=config.hidden_size, 
                               eps=config.rms_norm_eps, elementwise_affine=True)

    def forward(self, encoder_hidden_states, hidden_states, rope_rotations):
        self_attention_out_norm = self.norm1(self.self_attention(hidden_states, rope_rotations=rope_rotations))
        pre_cross_attention = self_attention_out_norm + hidden_states

        cross_attention_out_norm = self.norm2(self.cross_attention(encoder_hidden_states, hidden_states_decoder=pre_cross_attention))
        pre_mlp = cross_attention_out_norm + pre_cross_attention
        
        post_mlp_norm = self.norm3(self.mlp(pre_mlp))

        return post_mlp_norm + pre_mlp

def shift_right(input_ids, decoder_start_token_id, pad_token_id):
    shifted = input_ids.new_zeros(input_ids.shape)
    shifted[:, 1:] = input_ids[:, :-1]
    shifted[:, 0] = decoder_start_token_id
    shifted.masked_fill_(shifted == -100, pad_token_id)
    return shifted


class LanguageTransformerForCausalLM(PreTrainedModel, GenerationMixin):
    """A language model based on the Transformer architecture."""
    
    config_class = ModelConfig

    def __init__(self, config):
        super().__init__(config)

        self.rotary_emb = RotaryEmbedding(config)

        self.encoder_embedding = nn.Embedding(num_embeddings=config.vocab_size,
                                            embedding_dim=config.hidden_size)
        self.decoder_embedding = nn.Embedding(num_embeddings=config.vocab_size,
                                            embedding_dim=config.hidden_size)
        self.normalizer = nn.RMSNorm(config.hidden_size,
                                     eps=config.rms_norm_eps,
                                     elementwise_affine=True)
        self.unembedding = nn.Linear(in_features=config.hidden_size, out_features=config.vocab_size)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(config) for i in range(config.num_hidden_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(config) for i in range(config.num_hidden_layers)
        ])

        self.post_init()


    def forward(
        self,
        input_ids=None,
        decoder_input_ids=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        **kwargs
    ):
        # If decoder_input_ids are used then this means that we are in generation
        if decoder_input_ids is None:
            if labels is not None:
                decoder_input_ids = shift_right(
                    labels,
                    self.config.decoder_start_token_id,
                    self.config.pad_token_id,
                )
            else:
                raise ValueError("Need decoder_input_ids or labels")


        encoder_rope_rotations = self.rotary_emb(input_ids) # pass this to all the transformer decoder layers
        decoder_rope_rotations = self.rotary_emb(decoder_input_ids)

        encoder_embedding = self.encoder_embedding(input_ids)
        decoder_embedding = self.decoder_embedding(decoder_input_ids)

        for encoder in self.encoder_layers:
            encoder_embedding = encoder(encoder_embedding, encoder_rope_rotations)

        for decoder in self.decoder_layers:
            decoder_embedding = decoder(encoder_embedding, decoder_embedding, decoder_rope_rotations)
       
        decoder_out_norm = self.normalizer(decoder_embedding)
        logits = self.unembedding(decoder_out_norm)
       
        return Seq2SeqLMOutput(logits=logits)
        

#### RoPE implementation (copied and simplified from HuggingFace). ####

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

class RotaryEmbedding(nn.Module):
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