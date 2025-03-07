import torch
from transformers import MarianMTModel, MarianTokenizer
from torch.utils.data import DataLoader, TensorDataset
from transformers import Trainer, TrainingArguments

# Load pretrained MarianMT model and tokenizer
model_name = 'Helsinki-NLP/opus-mt-en-es'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Example data: inputs and labels (tokenized) from preprocessing
# Assume inputs and labels are tensorized as in preprocess.py

def train_model(inputs, labels):
    # Prepare data loader
    dataset = TensorDataset(inputs['input_ids'], labels)
    data_loader = DataLoader(dataset, batch_size=16)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=3,             # number of training epochs
        per_device_train_batch_size=16, # batch size for training
        per_device_eval_batch_size=16,  # batch size for evaluation
        warmup_steps=500,               # number of warmup steps for learning rate scheduler
        weight_decay=0.01,              # strength of weight decay
        logging_dir='./logs',           # directory for storing logs
        logging_steps=10,
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,  # Placeholder, should be a validation set
    )

    # Start training
    trainer.train()

# Example usage
train_model(inputs, labels)
