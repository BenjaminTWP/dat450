
import numpy as np 
import pandas as pd
import os

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

from langchain_community.llms import HuggingFacePipeline
from huggingface_hub import login

login()


def get_docs_quest():

    tmp_data = pd.read_json("ori_pqal.json").T
    # some labels have been defined as "maybe", only keep the yes/no answers
    tmp_data = tmp_data[tmp_data.final_decision.isin(["yes", "no"])]

    documents = pd.DataFrame({"abstract": tmp_data.apply(lambda row: (" ").join(row.CONTEXTS+[row.LONG_ANSWER]), axis=1),
                "year": tmp_data.YEAR})
    questions = pd.DataFrame({"question": tmp_data.QUESTION,
                "year": tmp_data.YEAR,
                "gold_label": tmp_data.final_decision,
                "gold_context": tmp_data.LONG_ANSWER,
                "gold_document_id": documents.index})

    return documents, questions

documents, questions = get_docs_quest()



hf = HuggingFacePipeline.from_model_id(
    model_id="meta-llama/Llama-3.2-1B",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 10},
)

from langchain_core.prompts import PromptTemplate

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

chain = prompt | hf

question = "What is electroencephalography?"

print(chain.invoke({"question": question}))


