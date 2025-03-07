import pandas as pd
from transformers import MarianTokenizer

# Function to load and preprocess the dataset
def preprocess_data(file_path):
    # Example: CSV file with 'English' and 'Spanish' columns
    df = pd.read_csv(file_path)

    # Tokenizer for MarianMT (for multilingual translation)
    tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-es')

    # Preprocess the text: Tokenize and align English-Spanish sentences
    english_text = df['English'].tolist()
    spanish_text = df['Spanish'].tolist()

    # Tokenizing the English and Spanish text
    inputs = tokenizer(english_text, padding=True, truncation=True, return_tensors="pt")
    labels = tokenizer(spanish_text, padding=True, truncation=True, return_tensors="pt").input_ids

    return inputs, labels

# Example usage
file_path = "medical_text.csv"  # Replace with your dataset path
inputs, labels = preprocess_data(file_path)
