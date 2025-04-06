# 🧠 Creating an NLP Data Loader

**⏱ Estimated Time:** 60 minutes  
**📌 Use Case:** Language Translation (Multilingual NLP)

---

## 📘 Overview
As an AI engineer working on a cutting-edge language translation project, you're tasked with bridging the communication gap between speakers of different languages. Central to the success of this endeavor is **high-quality data** — large corpora of bilingual sentences that form the foundation for your models.

In PyTorch, the `DataLoader` plays an indispensable role in managing large datasets. For NLP tasks, where sentence lengths vary significantly, the `DataLoader` efficiently batches these variable-length sequences, enabling diverse training examples and optimized GPU usage.

Batching also supports **data shuffling**, preventing memorization of sequence order and improving model generalization. NLP-specific preprocessing steps such as **tokenization**, **padding**, and **numericalization** are also handled efficiently through hooks like `collate_fn`.

---

## 🗂 Table of Contents
- ✅ Setup
- ✅ Installing Required Libraries
- ✅ Importing Required Libraries
- ✅ Dataset vs DataLoader
- ✅ Iterators Explained
- ✅ Custom Dataset & DataLoader
- ✅ Tokenization and Tensor Conversion
- ✅ Padding with Custom `collate_fn`
- ✅ Exercise: French Dataset Loader
- ✅ [Optional] German-English Translation Pipeline

---

## 🛠 Setup

### 📦 Installing Required Libraries
```bash
!pip install nltk
!pip install transformers==4.42.1
!pip install sentencepiece
!pip install spacy
!pip install numpy==1.26.0
!python -m spacy download en_core_web_sm
!python -m spacy download de_core_news_sm
!pip install torch==2.2.2 torchtext==0.17.2
!pip install torchdata==0.7.1
!pip install portalocker numpy pandas scikit-learn
```

---

## 📥 Importing Required Libraries
```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random

import torchtext
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from torch.nn.utils.rnn import pad_sequence
from torchdata.datapipes.iter import IterableWrapper

from typing import Iterable, List
```

---

## 📊 Dataset vs DataLoader
- **Dataset**: A collection of data samples, typically (input, label) pairs.
- **DataLoader**: Efficiently batches, shuffles, and loads data. Abstracts away loop iteration logic.

---

## 🔁 What is an Iterator?
- An object with `__iter__()` and `__next__()` methods.
- Helps loop through large data without loading everything into memory.
- All **DataLoaders are iterators** in PyTorch.

---

## 📚 Custom Dataset & DataLoader
```python
sentences = [
    "If you want to know what a man's like...",
    "Fame's a fickle friend, Harry.",
    "It is our choices, Harry...",
    "Soon we must all face...",
    "Youth can not know how age thinks...",
    "You are awesome!"
]

class CustomDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]
```

---

## 🔡 Tokenization and Tensor Conversion
```python
tokenizer = get_tokenizer("basic_english")
vocab = build_vocab_from_iterator(map(tokenizer, sentences))

class CustomTensorDataset(Dataset):
    def __init__(self, sentences, tokenizer, vocab):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.sentences[idx])
        return torch.tensor([self.vocab[token] for token in tokens])
```

---

## 🧰 Padding with Custom `collate_fn`
```python
def collate_fn(batch):
    return pad_sequence(batch, batch_first=True, padding_value=0)
```

---

## 🧪 Complete DataLoader Pipeline
```python
class RawTextDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

custom_dataset = RawTextDataset(sentences)

def collate_fn(batch):
    tensor_batch = [torch.tensor([vocab[token] for token in tokenizer(sample)]) for sample in batch]
    return pad_sequence(tensor_batch, batch_first=True)

dataloader = DataLoader(custom_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
```

---

## 🧩 Exercise: French Dataset Loader
```python
corpus = ["Ceci est une phrase.", "C'est un autre exemple...", ...]
corpus_sorted = sorted(corpus, key=lambda x: len(x.split()))
french_tokenizer = get_tokenizer("basic_english")
french_vocab = build_vocab_from_iterator(map(french_tokenizer, corpus_sorted))

def collate_french(batch):
    tensor_batch = [torch.tensor([french_vocab[token] for token in french_tokenizer(sentence)]) for sentence in batch]
    return pad_sequence(tensor_batch, batch_first=True)

french_dataset = RawTextDataset(corpus_sorted)
french_loader = DataLoader(french_dataset, batch_size=4, collate_fn=collate_french)

for batch in french_loader:
    print(batch)
```

---

## 🌐 [Optional] German-English Translation with Multi30k
## 📥 Load and Transform

```pythoon
multi30k.URL["train"] = "https://cf.../training.tar.gz"
multi30k.URL["valid"] = "https://cf.../validation.tar.gz"
SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
```
## 🔤 Tokenization Setup

```python
token_transform = {
    SRC_LANGUAGE: get_tokenizer('spacy', language='de_core_news_sm'),
    TGT_LANGUAGE: get_tokenizer('spacy', language='en_core_web_sm')
}
```
## 🔖 Special Symbols

```python
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
```

## 🔁 Yield Tokens

```python
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}
    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])
```

## 🧱 Build Vocabulary

```python
vocab_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    train_iterator = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    sorted_dataset = sorted(train_iterator, key=lambda x: len(x[0].split()))
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(sorted_dataset, ln), min_freq=1, specials=special_symbols, special_first=True)
    vocab_transform[ln].set_default_index(UNK_IDX)
```

## 🔁 Transform Pipelines
def tensor_transform_s(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]), torch.flip(torch.tensor(token_ids), dims=(0,)), torch.tensor([EOS_IDX])))

def tensor_transform_t(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX])))

def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

text_transform = {
    SRC_LANGUAGE: sequential_transforms(token_transform[SRC_LANGUAGE], vocab_transform[SRC_LANGUAGE], tensor_transform_s),
    TGT_LANGUAGE: sequential_transforms(token_transform[TGT_LANGUAGE], vocab_transform[TGT_LANGUAGE], tensor_transform_t)
}
```

## 🧪 Collate for Transformer Model

```python
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_seq = text_transform[SRC_LANGUAGE](src_sample.rstrip("\n"))
        tgt_seq = text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n"))
        src_batch.append(torch.tensor(src_seq, dtype=torch.int64))
        tgt_batch.append(torch.tensor(tgt_seq, dtype=torch.int64))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    return src_batch.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')), tgt_batch.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
```

## 🧪 Final DataLoaders

```python
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k

# Define batch size
BATCH_SIZE = 4

# Load and sort the training data
train_iterator = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
sorted_train_iterator = sorted(train_iterator, key=lambda x: len(x[0].split()))

# Create training DataLoader
train_dataloader = DataLoader(
    sorted_train_iterator,
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
    drop_last=True
)

# Load and sort the validation data
valid_iterator = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
sorted_valid_iterator = sorted(valid_iterator, key=lambda x: len(x[0].split()))

# Create validation DataLoader
valid_dataloader = DataLoader(
    sorted_valid_iterator,
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
    drop_last=True
)

# Preview a batch
src_batch, tgt_batch = next(iter(train_dataloader))
print("Source batch shape:", src_batch.shape)
print("Target batch shape:", tgt_batch.shape)
```

