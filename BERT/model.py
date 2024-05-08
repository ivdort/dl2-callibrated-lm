from transformers import BertConfig, BertForMaskedLM, BertTokenizer
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm


#tokenize
# tokenizer = BertTokenizer.from_pretrained('bert_tokenizer') # adjust to own path
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# Model config
config = BertConfig(
    vocab_size=tokenizer.vocab_size,  # needs to be adjusted to the dataset
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=7 # adjust this as well
)

model = BertForMaskedLM(config)


#data
class CustomDataset(Dataset):
    def __init__(self, sentences, tokenizer):
        self.sentences = sentences
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        # Tokenize the sentence and randomly mask some tokens
        inputs = self.tokenizer(sentence, padding='max_length', truncation=True, max_length=7, return_tensors='pt')
        input_ids = inputs["input_ids"].squeeze()
        labels = input_ids.clone()
        masked_indices = torch.randint(0, input_ids.size(0), input_ids.size(), dtype=torch.long)
        labels[masked_indices] = tokenizer.mask_token_id
        inputs = {"input_ids": input_ids, "labels": labels}
        return inputs

DATASET = 'math_dataset_1_to_10_+_-_ .csv'
df = pd.read_csv(f'C:/Users/lenna/Documents/UvA/DL2/dl2-callibrated-lm/dataset/{DATASET}') # adjust to own path
df['sentences'] = df['Equation'] + df['Answer'].astype(str)
sentences = [str(item) for item in df['sentences'].tolist()]  
train_sentences, test_sentences = train_test_split(sentences, test_size=0.2, random_state=42)
train_dataset = CustomDataset(train_sentences, tokenizer)
test_dataset = CustomDataset(test_sentences, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

#training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

num_epochs = 5
for epoch in range(num_epochs):
    # print('train')
    model.train()
    for _,batch in enumerate(tqdm(train_loader)):
        # print('training')
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        # print('loss')
        loss = outputs.loss
        optimizer.zero_grad()
        # print('backward')
        loss.backward()
        # print('optimize')
        optimizer.step()

    # Evaluation
    print('evaluation')
    model.eval()
    total_loss = 0
    num_examples = 0
    with torch.no_grad():
        for _,batch in enumerate(tqdm(test_loader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item() * batch["input_ids"].size(0)
            num_examples += batch["input_ids"].size(0)

    avg_loss = total_loss / num_examples
    print(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss}")

# save model
model.save_pretrained(f'saved_bert_model/{DATASET}/epochs_{num_epochs}')
