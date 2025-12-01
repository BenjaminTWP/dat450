from data import prepare_data

def section(title):
    print(f"\n################################ {title} ###############################\n", flush=True)

if __name__ == "__main__": 
    section("Step 1: Get the dataset")
    documents, question = prepare_data()