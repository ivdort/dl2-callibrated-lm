from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import BertTokenizer
import pandas as pd

# tokenizer init
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# trainer
trainer = WordPieceTrainer(
    vocab_size=2000,#30522,  # adjust!
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    limit_alphabet=1000,
    min_frequency=2
)

# data
df = pd.read_csv('C:/Users/lenna/Documents/UvA/DL2/dl2-callibrated-lm/dataset/math_dataset.csv') #adjust to own path on your system
df['sentences'] = df['Equation'] + df['Answer'].astype(str)
sentences = [str(item) for item in df['sentences'].tolist()]  
# actual training
tokenizer.train_from_iterator(sentences, trainer)

# saving
tokenizer.save("BERT/tokenizer")

vocab = tokenizer.get_vocab()
# merges = tokenizer.get_merges()

# Create a list of tokens and their indices
tokens = [token for token, _ in sorted(vocab.items(), key=lambda x: x[1])]
indices = {token: index for index, token in enumerate(tokens)}

# Create a vocab.txt file containing the list of tokens
with open("vocab.txt", "w", encoding="utf-8") as f:
    for token in tokens:
        f.write(token + "\n")

# Initialize a BertTokenizer using the vocab file
bert_tokenizer = BertTokenizer(vocab_file="vocab.txt", merges_file=None)

# Set the special tokens
bert_tokenizer.add_special_tokens({"additional_special_tokens": ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]})

# Save the tokenizer
bert_tokenizer.save_pretrained("bert_tokenizer")
