#load model and make predictions
import torch
from transformers import BertForMaskedLM, BertTokenizer

# load model
model = BertForMaskedLM.from_pretrained("saved_bert_model")

# tokenizer
tokenizer = BertTokenizer.from_pretrained("bert_tokenizer") # adjust to own path


def generate_text(input_ids):
    # Generate output predictions
    with torch.no_grad():
        outputs = model(input_ids)
        predictions = outputs[0]

    # Get the predicted token
    new_index = torch.argmax(predictions, -1)[:,-1].item()

    # Replace the masked token with the predicted token
    input_ids[0,-1] = new_index # index 5 -> after = sign
    generated_text = tokenizer.decode(input_ids.squeeze().tolist(), skip_special_tokens=True)

    return generated_text

import pandas as pd
import random


df = pd.read_csv('C:/Users/lenna/Documents/UvA/DL2/dl2-callibrated-lm/dataset/math_dataset.csv') # adjust to own path
df['sentences'] = df['Equation'] + df['Answer'].astype(str)
for _ in range(1):
    input = df['sentences'][random.randint(0, len(df)-1)]
    input_ids = tokenizer.encode(input, padding='max_length', truncation=True, max_length=7, return_tensors='pt')
    input_ids[0,-1] = tokenizer.mask_token_id
    print("input text:", input)
    # print("encoded input text:", input_ids)

    # input_text = "57 - 17 = [MASK]"
    generated_text = generate_text(input_ids)
    print("Generated text:", generated_text)