#load model and make predictions
import torch
from transformers import BertForMaskedLM, BertTokenizer

# load model
model = BertForMaskedLM.from_pretrained("saved_bert_model")

# tokenizer
# tokenizer = BertTokenizer.from_pretrained("bert_tokenizer") # adjust to own path
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def generate_text(input_ids):
    # Generate output predictions
    # labels = input_ids.clone()
    # labels[0,-2] = tokenizer.mask_token_id
    # inputs = {"input_ids": input_ids, "labels": labels}
    # batch = {k: v for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=torch.Tensor([[1,1,1,1,1,0,1]]))
        predictions = outputs[0]

    # Get the predicted token
    new_index = torch.argmax(predictions, -1)[:,-1].item()
    print(torch.argmax(predictions, -1))
    print(tokenizer.decode(torch.argmax(predictions, -1)[:,-2].item()))
    print(tokenizer.decode(new_index))
    print(input_ids)

    # Replace the masked token with the predicted token
    input_ids[0,-2] = new_index # index 5 -> after = sign
    generated_text = tokenizer.decode(input_ids.squeeze().tolist(), skip_special_tokens=True)

    return generated_text

import pandas as pd
import random

DATASET = 'math_dataset_1_to_10_+_-_ .csv'

df = pd.read_csv(f'C:/Users/lenna/Documents/UvA/DL2/dl2-callibrated-lm/dataset/{DATASET}') # adjust to own path
df['sentences'] = df['Equation'] + df['Answer'].astype(str)
for _ in range(1):
    input = df['sentences'][random.randint(0, len(df)-1)]
    input = [i for i in input.split(' ')[:-1]]
    input = ' '.join(c for c in input)
    input = input + ' [MASK]'
    print(input)
    input_ids = tokenizer.encode(input, padding='max_length', truncation=True, max_length=7, return_tensors='pt')
    # input_ids[0,-2] = tokenizer.mask_token_id
    print("input text:", input)
    print("encoded input text:", input_ids)

    generated_text = generate_text(input_ids)
    print("Generated text:", generated_text)