import pandas as pd
import sentencepiece as spm
from transformers import PreTrainedTokenizer
from datasets import Dataset, DatasetDict
import os

class CustomChatGLMTokenizer(PreTrainedTokenizer):
    def __init__(self, model_file):
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(model_file)

        self.vocab = {self.sp_model.id_to_piece(i): i for i in range(self.sp_model.get_piece_size())}
        self.ids_to_tokens = {i: self.sp_model.id_to_piece(i) for i in range(self.sp_model.get_piece_size())}
        
        # Set special tokens
        if '<pad>' in self.vocab:
            self.pad_token_id = self.vocab['<pad>']
        else:
            self.pad_token_id = self.sp_model.get_piece_size()  # If <pad> token does not exist, set to an unused ID
        
        self.padding_side = "right"

    def _tokenize(self, text):
        return self.sp_model.EncodeAsPieces(text)

    def _convert_token_to_id(self, token):
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        return self.sp_model.id_to_piece(index)

    def convert_tokens_to_ids(self, tokens):
        return [self._convert_token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self._convert_id_to_token(id) for id in ids]

    def encode(self, text, add_special_tokens=True, max_length=None, truncation=False, padding=False):
        tokens = self._tokenize(text)
        if max_length and truncation:
            tokens = tokens[:max_length]
        ids = self.convert_tokens_to_ids(tokens)
        if padding and max_length:
            ids = ids + [self.pad_token_id] * (max_length - len(ids))
        return ids

    def vocab_size(self):
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        return self.vocab

# Paths to the model file and dataset
model_file = "C:/Users/maham/DiagnoTechPC/FineTunedChatbot/ice_text.model"
csv_file = "C:/Users/maham/DiagnoTechPC/FineTunedChatbot/qna.csv"

# Ensure the paths exist
if not os.path.exists(model_file):
    print(f"Model file not found: {model_file}")
    exit(1)
if not os.path.exists(csv_file):
    print(f"CSV file not found: {csv_file}")
    exit(1)

# Load the CSV file
df = pd.read_csv(csv_file)

# Convert DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Split the dataset into train and validation sets
train_test_split = dataset.train_test_split(test_size=0.1)
datasets = DatasetDict({
    'train': train_test_split['train'],
    'validation': train_test_split['test']
})

print("Dataset columns:", datasets['train'].column_names)

# Initialize the custom tokenizer
tokenizer = CustomChatGLMTokenizer(model_file)

def tokenize_function(examples):
    questions = examples['Question']
    answers = examples['Answer']
    
    inputs = [f"Question: {q}\nAnswer: {a}" for q, a in zip(questions, answers)]
    
    tokenized = [tokenizer.encode(text, truncation=True, padding=True, max_length=1024) for text in inputs]
    
    return {'input_ids': tokenized}

# Apply tokenization to the dataset
tokenized_ds = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=datasets['train'].column_names,
    num_proc=4
)

# Set the format of the dataset to PyTorch tensors
tokenized_ds.set_format("torch")

print(tokenized_ds)

# Save the preprocessed dataset
# tokenized_ds.save_to_disk("./preprocessed_medical_dataset_chatglm")
