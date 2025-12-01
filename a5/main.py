from langchain_lm import create_pipeline, prompt
from data import prepare_data

def section(title):
    print(f"\n################################ {title} ###############################\n", flush=True)

if __name__ == "__main__": 
    section("Step 1: Get the dataset")
    documents, question = prepare_data()

    section("Step 2: Configure your LangChain LM")

    hf_pipeline = create_pipeline("Qwen/Qwen2.5-0.5B-Instruct")
    prompt("What is electroencephalography?", hf_pipeline)
