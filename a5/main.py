from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_lm import create_pipeline, prompt, rag_chain_prompt
from langchain_chroma import Chroma
from evaluate import Evaluator
from data import prepare_data
import numpy as np

############################ Some settings ############################

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

#######################################################################


def section(title):
    print(f"\n################################ {title} ###############################\n", flush=True)

def test_embedding_size(abstract, result):
    result = np.array(result)
    print(f"The embedding is of size is {result.shape}")

def example_text_chunks(texts):
    print("\nOne example text chunk is:\n", texts[4])
    print("\nAnother example text chunk is:\n", texts[5])


if __name__ == "__main__": 
    section("Step 1: Get the dataset")
    documents, questions = prepare_data()

    section("Step 2: Configure your LangChain LM")

    hf_pipeline = create_pipeline(MODEL_NAME)
    _ = prompt("What is electroencephalography?", hf_pipeline, sanity=True)

    section("Step 3: Set up the document database")

    embedding_func = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cuda"})
    query_result = embedding_func.embed_query("This seems to be a test")
    test_embedding_size("This seems to be a test", query_result)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)

    metadatas = [{"id": idx} for idx in documents.index]
    texts = text_splitter.create_documents(texts=documents.abstract.tolist(), metadatas=metadatas)
    example_text_chunks(texts)

    vector_store = Chroma(collection_name="abstracts",embedding_function=embedding_func)

    batch_size = 5000 
    for i in range(0, len(texts), batch_size):
        vector_store.add_documents(texts[i:i+batch_size])

    results = vector_store.similarity_search_with_score("What is programmed cell death?", k=3)

    print(f"\nExample question and some retrievals (question = 'What is programmed cell death?'):\n")
    for res, score in results:
        print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")

    section("Step 4: Define the full RAG pipeline (OPTION B)")

    retriever = vector_store.as_retriever(search_kwargs={'k':2})
    _, _, _ = rag_chain_prompt(questions.iloc[6].question, hf_pipeline, retriever, sanity=True)

    section("Step 5: Evaluate RAG on the dataset")

    evaluator = Evaluator(hf_pipeline, documents, questions, retriever, metadatas)
    evaluator.run_evaluation()
