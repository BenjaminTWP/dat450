import pandas as pd

data_loc = "/data/users/benpe/downloads/data/pubmed/ori_pqal.json"

def prepare_data():
    tmp_data = pd.read_json(data_loc).T
    # some labels have been defined as "maybe", only keep the yes/no answers
    tmp_data = tmp_data[tmp_data.final_decision.isin(["yes", "no"])]

    documents = pd.DataFrame({"abstract": tmp_data.apply(lambda row: (" ").join(row.CONTEXTS+[row.LONG_ANSWER]), axis=1),
                "year": tmp_data.YEAR})
    questions = pd.DataFrame({"question": tmp_data.QUESTION,
                "year": tmp_data.YEAR,
                "gold_label": tmp_data.final_decision,
                "gold_context": tmp_data.LONG_ANSWER,
                "gold_document_id": documents.index})

    sanity_check(documents, questions)
    
    return documents, questions

def sanity_check(documents, questions):
    print("Example data:\n")
    print("Question:\n", questions.iloc[6].question)
    print("Abstract:\n", documents.iloc[6].abstract)