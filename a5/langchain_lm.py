from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import PromptTemplate


def create_pipeline(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model,
                    device=0, tokenizer=tokenizer,
                    return_full_text=False, max_new_tokens=32)

    hf = HuggingFacePipeline(pipeline=pipe)
    return hf

def prompt(question, hf):
    template = """Answer the following question with a yes or no. The answer must contain either "Answer: yes" or "Answer: no", not both. \n 
                  Question: {question}
                  """
    prompt = PromptTemplate.from_template(template)

    chain = prompt | hf

    print("\nThe question: \n", question)

    answer = ""
    for chunk in chain.stream({"question": question}):
        print(chunk, end="", flush=True)
        answer += chunk 

    return answer.strip() 


def rag_chain_prompt(question, hf_pipeline, retriever, sanity=False):
    template = """Answer the following question with a yes or no. The answer must contain either "Answer: yes" or "Answer: no", not both. \n
                  Question: {question}
                  Context:
                  {context}
                  """
    prompt_template = PromptTemplate.from_template(template)
    
    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])
    
    prompt_text = prompt_template.format(question=question, context=context)

    if sanity:
        print("\nPrompt with context: \n", prompt_text)
    
    chain = ( prompt_template | hf_pipeline | StrOutputParser() )
    
    runnable_parallel_object = RunnableParallel({"answer": chain})
    
    results = runnable_parallel_object.invoke({
        "question": question,
        "context": context
    })

    if sanity:
        print("\nThe answer: \n", results["answer"])

    return results["answer"]