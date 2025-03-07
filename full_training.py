from transformers import MarianMTModel, MarianTokenizer
from torch.optim import Adam
from preprocess import preprocess_data
from fine_tune import train_model
from rlhf import rlhf_update

# Load data (Assume data file with English-Spanish parallel medical text)
file_path = 'medical_text.csv'
inputs, labels = preprocess_data(file_path)

# Initialize model and optimizer
model_name = 'Helsinki-NLP/opus-mt-en-es'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
optimizer = Adam(model.parameters(), lr=5e-6)

# Step 1: Fine-tune with standard data
train_model(inputs, labels)

# Step 2: RLHF-based training
english_text = "Patient has a headache."
spanish_text = "El paciente tiene dolor de cabeza."
feedback = 0.9  # Simulated human feedback score
rlhf_update(model, english_text, spanish_text, feedback, optimizer)
