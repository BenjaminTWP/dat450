from a2.A2_skeleton import (
    A2ModelConfig,
    A2MLP,
    A2Attention,
    A2RotaryEmbedding,
    A2DecoderLayer,
    A2Transformer
)
import torch

#import os, sys
#PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#sys.path.append(PROJECT_ROOT)
#print("CWD:", os.getcwd())
#print("sys.path:", sys.path)


from a1.A1_skeleton import (
    A1Tokenizer,
    A1Trainer,
)

hidden_size = 128
vocab_size = 150000

config = A2ModelConfig(
    vocab_size=vocab_size,
    hidden_size=hidden_size,
    max_position_embeddings=100000,
    rms_norm_eps=0.001,
    num_attention_heads=2,
    rope_theta=2,
    hidden_act='silu',
    intermediate_size=64,
    num_hidden_layers=2,
)

tensor = torch.ones((5, 20, hidden_size)) * 100

mlp_model = A2MLP(config)
out = mlp_model(tensor)

print("Check shape match after MLP:", out.shape == tensor.shape)

rotary_model = A2RotaryEmbedding(config)
rotation_emb = rotary_model(tensor)

attention_model = A2Attention(config)
out = attention_model(tensor, rotation_emb)

print("Check shape match after attention:", out.shape == tensor.shape)

decoder = A2DecoderLayer(config)
out = decoder(tensor, rotation_emb)
print("Check shape match after decoder:", out.shape == tensor.shape)


TOKENIZER = "a2/tokenizer.pkl"
tokenizer = A1Tokenizer.from_file(TOKENIZER)
transformer = A2Transformer(config)

text = ["Hello my name is jeff and"]
emb_tensor = tokenizer(text, return_tensors='pt', padding=True, truncation=True).input_ids
out = transformer(emb_tensor)

# Change shape of the tensor to be (1,7, vocab_size)
emb_tensor = emb_tensor.unsqueeze(-1)
emb_tensor = emb_tensor.expand(-1, -1, vocab_size) 
print("Check shape match after transformer:", out.shape == emb_tensor.shape)

#from a2.run_prediction import predict
#from argparse import Namespace

#args = Namespace(
#    temperature=0,
#    generation_max_length=100,
#    npreds=10,
#    output_dir="a2/trainer_output"
#)


p#redict(transformer, tokenizer, args)

