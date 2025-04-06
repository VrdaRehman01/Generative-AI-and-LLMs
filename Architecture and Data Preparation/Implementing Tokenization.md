# 📘 Tokenization in Natural Language Processing (NLP)

## 📌 Overview

Tokenizers are essential tools in NLP that break text into smaller units called tokens. These can be words, characters, or subwords. This segmentation makes text comprehensible to machines and powers applications like translation, sentiment analysis, and chatbots.

---

## 📋 Table of Contents

1. Objectives  
2. Setup & Required Libraries  
3. What is a Tokenizer?  
4. Types of Tokenizers  
   - Word-based  
   - Character-based  
   - Subword-based  
5. WordPiece  
6. Unigram & SentencePiece  
7. Tokenization with PyTorch  
8. Token Indices & OOV Handling  
9. Practical Implementation  
10. Exercise: Comparative Tokenization  

---

## 🎯 Objectives

After completing this section, you will be able to:

- Understand tokenization and its role in NLP.  
- Identify and differentiate tokenization methods.  
- Apply tokenization in Python using various libraries.  
- Map tokens to indices for machine learning models.  

---

## ⚙️ Setup & Required Libraries

### ✅ Install dependencies:

```bash
!pip install nltk transformers==4.42.1 sentencepiece spacy torch==2.2.2 torchtext==0.17.2
!python -m spacy download en_core_web_sm
```

---


#### Import Libraries 
```python
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")

import spacy
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.util import ngrams
from transformers import BertTokenizer, XLNetTokenizer
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
```

---

# 🧠 What is a Tokenizer?

Tokenizers segment raw text into smaller units (tokens) and convert them into numerical values (token indices). These indices feed into deep learning models.

## 📊 Diagram: Text → Tokenization → Tokens
**Example:**  
`IBM taught me tokenization` → `["IBM", "taught", "me", "tokenization"]`

---

## 🔍 Types of Tokenizers

### 📊 Diagram: Tokenization Methods  
**Tokenization methods:**  
- **Word-based**  
- **Character-based**  
- **Subword-based**

## 1. Word-Based Tokenizer

Splits text into words based on whitespace or punctuation.

**🛠️ Tools:** `nltk`, `spaCy`

```python
from nltk.tokenize import word_tokenize

text = "This is a sample sentence for word tokenization."
tokens = word_tokenize(text)
print(tokens)
```

### 🔍 spaCy Example

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("I couldn't help the dog. Can't you do it? Don't be afraid.")

token_list = [token.text for token in doc]
print("Tokens:", token_list)

for token in doc:
    print(token.text, token.pos_, token.dep_)
```
> Note: Libraries tokenize contractions differently
`(e.g., "don't" → "do", "n't")`

⚠️ Drawback: Words like `unicorn` and `unicorns` are treated as distinct tokens.

---

## 2. Character-Based Tokenizer

Splits input into individual characters.

✅ **Advantage:** Smaller vocabulary  
⚠️ **Disadvantage:** More tokens, less semantic meaning

**Example:**

```text
Input: "Tokenization"
Output: ['T', 'o', 'k', 'e', 'n', 'i', 'z', 'a', 't', 'i', 'o', 'n']
```

---

## 3. Subword-Based Tokenizer

Combines the benefits of word-based and character-based tokenization.

### 🧠 Common Algorithms:
- **WordPiece**
- **Unigram**
- **SentencePiece**

### 📊 Diagram: Subword Examples
- `Unhappiness` → `Un`, `Happiness`  
- `Unicorns` → `Unicorn`, `s`

### 🔹 WordPiece (BERT)

WordPiece initializes the vocabulary with characters and learns merge rules that improve the likelihood of the training data.

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize("IBM taught me tokenization.")
print(tokens)
```

>ization is a subword attached to the previous token (`token`).

---

### 🔹 Unigram & SentencePiece (XLNet)

- **SentencePiece** segments text and assigns IDs consistently, treating input as a raw string (no whitespace-based pretokenization).
- **Unigram** reduces the vocabulary iteratively by removing less likely subword units.

```python
from transformers import XLNetTokenizer

tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
tokens = tokenizer.tokenize("IBM taught me tokenization.")
print(tokens)
```
Example Output:
`['▁IBM', '▁taught', '▁me', '▁token', 'ization', '.']`

## 🔧 Tokenization with PyTorch

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
>This approach integrates tokenization and vocabulary building for use in PyTorch pipelines.

---

## 🔁 Token Indices & OOV

OOV (Out-of-Vocabulary) words are mapped to `<unk>`.

```python
def get_tokenized_sentence_and_indices(iterator):
    tokenized_sentence = next(iterator)
    token_indices = [vocab[token] for token in tokenized_sentence]
    return tokenized_sentence, token_indices
```

## 🧪 Practical Tokenization Example

```python
lines = ["IBM taught me tokenization", "Special tokenizers are ready"]
tokenizer_en = get_tokenizer('spacy', language='en_core_web_sm')

special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
tokens = []
max_length = 0

for line in lines:
    tok_line = ['<bos>'] + tokenizer_en(line) + ['<eos>']
    tokens.append(tok_line)
    max_length = max(max_length, len(tok_line))

for i in range(len(tokens)):
    tokens[i] += ['<pad>'] * (max_length - len(tokens[i]))

vocab = build_vocab_from_iterator(tokens, specials=special_symbols)
vocab.set_default_index(vocab["<unk>"])
```
This example uses a `spaCy` tokenizer, adds special tokens like `<bos>` (beginning of sentence), `<eos>` (end of sentence), and applies padding to equalize sequence lengths.

---

## ⚖️ Exercise: Comparative Tokenization

Compare the performance of the following tokenizers:

- `nltk`
- `spaCy`
- `BertTokenizer`
- `XLNetTokenizer`

Measure tokenization time using `datetime` and analyze token frequencies with `FreqDist`.

```python
from datetime import datetime
from nltk import FreqDist

start = datetime.now()
# run tokenizer here
print("Time elapsed:", datetime.now() - start)
```

---

## ✅ Summary

- Tokenizers split text into processable units for AI models.

### 🔧 Techniques
- Word-based  
- Character-based  
- Subword-based

### 🧰 Tools
- `nltk`  
- `spaCy`  
- Hugging Face (`transformers`)  
- `torchtext`

### 🧪 Subword Algorithms
- WordPiece  
- SentencePiece  
- Unigram

### ⚙️ PyTorch Integration
- Uses `get_tokenizer` and `build_vocab_from_iterator`

### 🔖 Special Tokens
- `<bos>` (beginning of sentence)  
- `<eos>` (end of sentence)  
- `<pad>` (padding)  
- `<unk>` (unknown/OOV)

### 🔁 Additional Notes
- OOV words are handled using `<unk>`  
- Padding ensures uniform sequence lengths for batch processing
