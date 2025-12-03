
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
from langchain_core.prompts import PromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from uuid import uuid4
from typing import Any
from langchain_core.documents import Document
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents import create_agent


login(token="YOUR TOKEN")


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
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 1},
)



template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

chain = prompt | hf

question = "What is electroencephalography?"

print(chain.invoke({"question": question}))


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

text = "This is a test document."

query_result = embeddings.embed_query(text)

print(query_result[:3])

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    length_function=len,
    is_separator_regex=False,
)

metadatas = [{"id": idx} for idx in documents.index]
texts = text_splitter.create_documents(texts=documents.abstract.tolist(), metadatas=metadatas)

# for text in texts[:10]:
#     print(text)

# doc_result = embeddings.embed_documents([text])


vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
)

uuids = [str(uuid4()) for _ in range(len(texts))]

print(uuids[:10])

vector_store.add_documents(documents=texts, ids=uuids)

results = vector_store.similarity_search_with_score(
    "What is programmed cell death?", k=3
)
for res, score in results:
    print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")


print("_____________RAG_PIPELINE_____________")


class State(AgentState):
    context: list[Document]


class RetrieveDocumentsMiddleware(AgentMiddleware[State]):
    state_schema = State

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def before_model(self, state: AgentState) -> dict[str, Any] | None:
        last_message = state["messages"][-1] # get the user input query
        retrieved_docs = self.vector_store.similarity_search(last_message.text)  # search for documents

        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)  

        augmented_message_content = (
            "You are an expert assistant. Use the following context in your response:"
            f"\n\n{docs_content}"
            "\nAwnser the question by writing Awnser=Yes if the question is true and write Awnser=No if the question is false."
            "\nNow awnser the question, Answer="
        )
        return {
            "messages": [last_message.model_copy(update={"content": augmented_message_content})],
            "context": retrieved_docs,
        }


model = hf
agent = create_agent(model, tools=[], middleware=[RetrieveDocumentsMiddleware(vector_store)])

your_query = "Can people die due to cancer?"

for step in agent.stream(
    {"messages": [{"role": "user", "content": your_query}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()


