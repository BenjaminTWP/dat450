from A1_skeleton import build_tokenizer, TrainingArguments, A1Trainer, A1Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from A2_skeleton import *
from torch import nn
import numpy as np

train_file_loc = "/data/courses/2025_dat450_dit247/assignments/a1/train.txt"
val_file_loc = "/data/courses/2025_dat450_dit247/assignments/a1/val.txt"
local_dir = '/data/courses/2025_dat450_dit247/models/OLMo-2-0425-1B'

def get_test_config():
    vocab_size = len(A1Tokenizer.from_file("vocab"))

    return A2ModelConfig(vocab_size=vocab_size, hidden_size=64,
                         intermediate_size=16, num_attention_heads=4, 
                         num_hidden_layers=8, rope_theta=10000.0,
                         hidden_act='silu', max_position_embeddings=None, 
                         rms_norm_eps=0.1)

def get_train_config():
    return TrainingArguments(lr=1e-3, epochs=2, batch_size=8)

def sanity_mlp_rms():
    config = get_test_config()
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

def sanity_transformer():
    config = get_test_config()
    transformer = A2Transformer(config)

    intgrs = np.random.randint(low=0, high=500, size=(5, 10)).tolist()
    int_tensor = torch.tensor(intgrs)

    print(f"#\nFeeding a two-dimensional tensor to the transformer, tensor shape before {int_tensor.shape}")

    vec_out = transformer(int_tensor)
    print(f"#\nOutput of the vector after the transformer {vec_out.shape}")


def train_transformer():
    dataset = load_dataset('text', data_files={'train': train_file_loc, 'val': val_file_loc})
    dataset = dataset.filter(lambda x: x['text'].strip() != '')

    tokenizer = A1Tokenizer.from_file("vocab")

    config = get_test_config()

    transformer = A2Transformer(config)

    args = get_train_config()

    a1_trainer = A1Trainer(transformer, args, dataset['train'], dataset['val'], tokenizer)

    a1_trainer.train()

    return transformer

def test_some_sentences(model):
    tokenizer = A1Tokenizer.from_file("vocab")

    sentences = ["he lives in San", 
                 "The capital of ",
                 "The fifth element in the periodic table is"]

    for sentence in sentences:
        print(f"The 5 best results for following sentence '{sentence}' is: \n")

        encoding = tokenizer([sentence], return_tensors='pt')
        output = model(encoding['input_ids'])
        topk = torch.topk(output[0, -2], k=5)

        for idx, score in zip(topk.indices, topk.values):
            
            print(tokenizer.int_to_str[idx.item()], float(score.detach()))
        
        print("\n")

def random_sampling(model, prompt, max_length, temperature, topk):
    pass


def test_text_generation(model):
    prompts = ['In natural language processing, a Transformer'
                'Is Stockholm the capital of Sweden? Answer yes or no. The answer is'
                'Write a Python program that reverses a list.']

    for prompt in prompts:
        random_sampling(model, prompt, max_length=7, topk=4)


def load_local_model():
    tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(local_dir, local_files_only=True)
    print(model)


if __name__ == "__main__":
    print("#", flush=True)
    #sanity_mlp_rms()
    #sanity_transformer()

    transformer = train_transformer()
    #trained_transformer = A2Transformer.from_pretrained("trained_model")
    #test_some_sentences(trained_transformer)