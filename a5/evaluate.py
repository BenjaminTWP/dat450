from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from langchain_lm import prompt, rag_chain_prompt
import numpy as np
import time
import re

np.random.seed(10)

class Evaluator:
    def __init__(self, hf_pipeline, docs, questions, retriever, metadata):
        self.hf_pipeline = hf_pipeline
        self.docs = docs
        self.questions = questions
        self.retriever = retriever
        self.metadata = metadata

    def run_evaluation(self):
        random_examples = np.random.randint(0, high=len(self.questions), size=3)

        lm_answers = []
        rag_answers = []
        true_label = []
        gold_doc_freq = 0

        for index in range(len(self.questions)):
            lm_answer = prompt(self.questions.iloc[index].question, self.hf_pipeline)
            lm_label = self.label_answer(lm_answer)
            rag_answer, doc_ids, context = rag_chain_prompt(self.questions.iloc[index].question, self.hf_pipeline, self.retriever)
            rag_label = self.label_answer(rag_answer)

            if not lm_label == -1 and not rag_label == -1:
                lm_answers.append(lm_label)
                rag_answers.append(rag_label)
                true_label.append(self.question_gold_label(self.questions.iloc[index].gold_label))
            
            gold_doc_id = int(self.questions.iloc[index].gold_document_id)
            if gold_doc_id in doc_ids:
                gold_doc_freq += 1
            
            if index in random_examples:
                self.print_example(self.questions.iloc[index].question, context, lm_answer, rag_answer)            
        
        self.print_statistics(lm_answers, rag_answers, true_label)
        print(f"\nThe gold document is retrieved with a frequency of {gold_doc_freq/len(self.questions)}")

    def label_answer(self, answer):
        '''
        1  -  Answer is yes
        0  -  Answer is no
        -1 -  Answer is not found
        '''
        answer = answer.lower()
        
        to_find = "answer"
        to_find_len = len(to_find)

        positions = [match.start() for match in re.finditer(to_find, answer)]

        if not positions:
            return -1
        else:
            for position in positions:
                start = position + to_find_len
                end = start + 8
                search = answer[start: end]
                
                if "yes" in search:
                    return 1
                elif "no" in search:
                    return 0

        return -1

    def question_gold_label(self, gold):
        gold = gold.lower()

        if "yes" in gold:
            return 1
        else:
            return 0
        
    def print_statistics(self, lm_pred, rag_pred, true_label):
        lm_acc = accuracy_score(true_label, lm_pred)
        lm_rec = recall_score(true_label, lm_pred)
        lm_pre = precision_score(true_label, lm_pred)
        lm_f1  = f1_score(true_label, lm_pred)

        rag_acc = accuracy_score(true_label, rag_pred)
        rag_rec = recall_score(true_label, rag_pred)
        rag_pre = precision_score(true_label, rag_pred)
        rag_f1  = f1_score(true_label, rag_pred)

        print(f"\nFor the LM-model:  Acc={lm_acc:.2f}, Recall={lm_rec:.2f}, "
              f"Precision={lm_pre:.2f}, F1={lm_f1:.2f}")

        print(f"\nFor the RAG-model: Acc={rag_acc:.2f}, Recall={rag_rec:.2f}, "
              f"Precision={rag_pre:.2f}, F1={rag_f1:.2f}")

    def print_example(self, question, context, lm_answer, rag_answer):
        print("\n=== Example ===")
        print("Question:", question)

        print("\nRetrieved documents (context):", context)

        print("\nLM Answer:", lm_answer)

        print("\nRAG Answer:", rag_answer)
        print("================\n")