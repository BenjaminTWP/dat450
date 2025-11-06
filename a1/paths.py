from pathlib import Path

BASE_PATH = str(Path(__file__).parent)

TRAIN_FILE = BASE_PATH + "/train.txt"
VAL_FILE = BASE_PATH + "/val.txt"
TRAINER_OUTPUT = BASE_PATH + "/trainer_output"
TOKENIZER = BASE_PATH + "/tokenizer.pkl"
