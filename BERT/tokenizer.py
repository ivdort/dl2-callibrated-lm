from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
import pandas as pd

# tokenizer init
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# trainer
trainer = WordPieceTrainer(
    vocab_size=30522,  # adjust!
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    limit_alphabet=1000,
    min_frequency=2
)

# data
df = pd.read_csv('/Users/devindewilde/Documents/GitHub/dl2-callibrated-lm/dataset/math_dataset.csv') #adjust to own path on your system
text1 = df['Equation'].tolist()
text2 = df['Answer'].tolist()
texts = text1 + text2
string_texts = [str(item) for item in texts]
# actual training
tokenizer.train_from_iterator(string_texts, trainer)

# saving
tokenizer.save("tokenizer.json")