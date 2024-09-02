import torch
from transformers import Trainer, TrainingArguments
from datasets import load_from_disk
from model import setup_model
from preprocess import tokenized_ds

# Another way to load the tokenized dataset
# tokenized_ds = load_from_disk("./tokenized_dataset")

# Load the model and tokenizer
model, tokenizer = setup_model()

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    gradient_accumulation_steps=4,
    fp16=True,  # if GPU supports it
    learning_rate=5e-5,
    max_grad_norm=1.0,
)

# Define a custom data collator
def data_collator(features):
    return {
        "input_ids": torch.stack([f["input_ids"] for f in features]),
        "attention_mask": torch.stack([f["attention_mask"] for f in features]),
        "labels": torch.stack([f["input_ids"] for f in features]),
    }

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    data_collator=data_collator,
)

# Start training
trainer.train()

# Save the fine-tuned model
trainer.save_model("./fine_tuned_model")