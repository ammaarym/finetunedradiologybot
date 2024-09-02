from transformers import AutoModel, AutoTokenizer
import torch

def setup_model():
    # Load the model and tokenizer
    model_name = "THUDM/chatglm-6b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    # # Move model to GPU if available
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    # Set model to training mode
    model.train()

    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = setup_model()
    print(f"Model loaded: {type(model)}")
    print(f"Tokenizer loaded: {type(tokenizer)}")
    print(f"Model device: {next(model.parameters()).device}")