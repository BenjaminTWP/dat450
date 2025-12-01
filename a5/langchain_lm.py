from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.prompts import PromptTemplate


def create_pipeline(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model,
                    device=0, tokenizer=tokenizer,
                    return_full_text=False, max_new_tokens=128)

    hf = HuggingFacePipeline(pipeline=pipe)
    return hf

def prompt(question, hf):
    template = """Question: {question}"""
    prompt = PromptTemplate.from_template(template)

    chain = prompt | hf

    print("\nThe question: \n", question)

    for chunk in chain.stream({"question": question}):
        print(chunk, end="", flush=True)