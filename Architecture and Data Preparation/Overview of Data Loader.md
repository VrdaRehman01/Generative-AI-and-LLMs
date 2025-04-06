# 📦 NLP with PyTorch: Data Loaders & Preprocessing

---

## 🎯 Learning Objectives

By the end of this lesson, you will be able to:

- Define what a **data loader** is  
- Explain its **purpose** in training NLP models  
- Understand the **PyTorch `DataLoader` class** and how **batching** works  

---

## 🧠 Why Use Data Loaders?

When working on large NLP tasks like machine translation, manually loading and shuffling datasets becomes inefficient. A `DataLoader` handles this automatically:

- Efficiently loads and preprocesses data  
- Enables **batching** and **shuffling**  
- Performs **on-the-fly preprocessing**  
- Optimizes **memory usage**  
- Seamlessly integrates with **PyTorch** training pipelines  

---

## ⚙️ Dataset Structure for NLP

Datasets are typically split into:

- **Training set** – Trains the model  
- **Validation set** – Tunes hyperparameters  
- **Test set** – Evaluates generalization  

Handled via:
- `Dataset` class  
- `DataLoader` class

---

## 🧩 Creating a Custom Dataset

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

# Usage
sentences = ["This is correct.", "It incorrect sentence.", "He are happy."]
dataset = CustomDataset(sentences)
print(dataset[0])  # Access the first sample
```

## 📦 Using DataLoader
The DataLoader class helps in batching, shuffling, and iterating through datasets:

```python
from torch.utils.data import DataLoader

data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

for batch in data_loader:
    print(batch)
```

## 🔄 What is an Iterator?

- An object that lets you loop over data using `iter()` and `next()`
- Enables efficient processing of large datasets

---

## 🔄 Transformations in NLP Pipelines

Typical preprocessing steps:

1. **Tokenization**  
   Splitting text into words, subwords, or characters.

2. **Numericalization**  
   Mapping tokens to numerical indices using a vocabulary.

3. **Padding**  
   Ensuring uniform input length by adding special tokens.

4. **Tensor Conversion**  
   Converting data into tensors for model compatibility.

---

## 🔡 Tokenization + Vocabulary Building

- **Tokenization** breaks text into smaller units (tokens).
- **Vocabulary Building** creates a mapping from tokens to indices.
- Helps models understand and process textual input efficiently.

```python
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(dataset), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
```

## 📏 Padding Sequences

```python
from torch.nn.utils.rnn import pad_sequence
import torch

sequences = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])]
padded = pad_sequence(sequences, batch_first=True, padding_value=0)
print(padded)
```
>Padding ensures that all sequences in a batch have equal length — critical for batching in training.

## 🧰 Custom `collate_fn` in PyTorch

Used to customize how batches are formed in the `DataLoader`.

### 🔧 Example: Custom Collate Function

```python
def collate_fn(batch):
    tokenized = [tokenizer(text) for text in batch]
    indexed = [[vocab[token] for token in tokens] for tokens in tokenized]
    tensor_batch = [torch.tensor(seq) for seq in indexed]
    padded = pad_sequence(tensor_batch, batch_first=True, padding_value=0)
    return padded
```
## 🔄 NLP Preprocessing Steps Explained

- **Tokenization**  
  Converts raw text into smaller units called *tokens* (e.g., words or subwords).

- **Indexing**  
  Maps each token to a unique numerical index using a predefined *vocabulary*.

- **Tensor Conversion**  
  Transforms the list of indexed sequences into PyTorch *tensors* for model input.

- **Padding**  
  Adds special tokens (usually `0`) to sequences to ensure a *uniform batch shape*.


##🔁 DataLoader with `collate_fn`
```pyhton
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
```
This enables dynamic preprocessing for each batch, especially useful in NLP tasks with variable-length sequences.


## ✅ Recap

- **Automates** data batching, shuffling, and loading  
- **Reduces memory usage** with lazy loading  
- **Allows dynamic transformation** using `collate_fn`  
- **Essential** for training large-scale NLP models efficiently

