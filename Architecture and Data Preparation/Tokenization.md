# ✂️ Tokenization in Natural Language Processing (NLP)

Tokenization is a foundational step in NLP that breaks a sentence into smaller components called **tokens**. These tokens are essential for AI models to understand and process human language effectively.

---

## 🎯 Learning Objectives

By the end of this module, you will be able to:
- Define tokenization and its purpose in NLP.
- Explain different tokenization methods and their characteristics.
- Identify how popular tokenizers (like NLTK, spaCy, and Hugging Face) operate.
- Understand and apply tokenization and indexing using PyTorch and `torchtext`.

---

## 📖 What is Tokenization?

Tokenization involves dividing input text into smaller pieces called **tokens**, which can be words, subwords, or characters.

For example:
```text
Input:  "IBM taught me tokenization"
Output: ["IBM", "taught", "me", "tokenization"]
```

---


These tokens help the model extract meaning, classify sentiment, and perform other NLP tasks.

---

## ⚙️ What is a Tokenizer?

- A **tokenizer** is a program that performs tokenization.
- It breaks the input text into meaningful units based on the selected method.

---

## 🔍 Types of Tokenization

### 1. **Word-Based Tokenization**
- Each word is treated as a token.
- **Advantage**: Preserves semantic meaning.
- **Disadvantage**: Expands vocabulary size significantly.

🧠 Note:
> Tools like **NLTK** and **spaCy** tokenize sentences but treat similar words (`unicorn` vs. `unicorns`) as different, which can cause issues in NLP tasks.

---

### 2. **Character-Based Tokenization**
- Each character is treated as a token.
- **Advantage**: Smaller vocabulary.
- **Disadvantage**: Lacks semantic meaning; increases computational load.

🧾 Example:
```text
Input:  "This is a sentence"
Output: ["T", "h", "i", "s", " ", "i", "s", " ", "a", " ", "s", "e", ...]
```
---

### 3. Subword-Based Tokenization

Frequently used words remain whole, while rare or unknown words are split into meaningful subword units.

**✅ Advantage:**  
Combines the benefits of word-based and character-based tokenization — maintaining semantic meaning while reducing vocabulary size.

---

### 🧠 Common Algorithms

#### 🔹 WordPiece
- Merges or splits symbols based on their utility.
- Used in models like BERT.

#### 🔹 Unigram
- Begins with a large vocabulary of potential subwords.
- Iteratively removes the least frequent subwords.
- Aims to minimize overall encoding loss.

#### 🔹 SentencePiece
- Treats text as a raw byte or character sequence.
- Segments the text into subword units.
- Assigns unique IDs to each segment.
- Can be used without whitespace pre-tokenization.

---

## 🧪 Implementation Examples with Hugging Face

### 🔹 WordPiece using BERT Tokenizer

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize("Tokenization using WordPiece")
```
>ization indicates that the token is a subword and should be attached to the previous token without a space.


---

### 🔹 Unigram and SentencePiece using XLNet

```python
from transformers import XLNetTokenizer

tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
tokens = tokenizer.tokenize("Tokenization using SentencePiece")
```
>Tokens with _ indicate that the word is preceded by a space in the original text.

---

## 🧰 Tokenization in PyTorch Using `torchtext`

PyTorch provides the `torchtext` library to tokenize text and build a vocabulary for NLP tasks.

---

### ✅ Step-by-Step Guide

#### 1. Tokenizer Setup

```python
from torchtext.data.utils import get_tokenizer

tokenizer = get_tokenizer("basic_english")
```

from torchtext.vocab import build_vocab_from_iterator

vocab = build_vocab_from_iterator(yield_tokens(dataset))
vocab.set_default_index(vocab["<unk>"])


---

#### 4. Map Tokens to Indices

```python
tokens = tokenizer("This is a test")
indices = [vocab[token] for token in tokens]
```

---

## 🔄 Tokenization + Indexing in Practice

### Example Function

```python
def get_tokenized_sentence_and_indices(data_iter):
    tokenized_sentence = next(data_iter)
    tokens = tokenizer(tokenized_sentence)
    indices = [vocab[token] for token in tokens]
    return tokens, indices
```
>Use vocab.get_stoi() to retrieve the word-to-index mapping.
The tokenizer processes each sentence individually and maps words to numerical indices.

---

## 🔖 Adding Special Tokens (spaCy Example)

You can manually add special tokens such as:

- `<BOS>` — Beginning of Sentence  
- `<EOS>` — End of Sentence

```python
tokens = ["<BOS>"] + tokenizer(sentence) + ["<EOS>"]
```

## 📏 Padding Tokenized Sentences

To ensure uniform input length for batching, pad tokens can be added:

```python
max_length = max(len(s) for s in tokenized_sentences)
padded_sentences = [s + ["<PAD>"] * (max_length - len(s)) for s in tokenized_sentences]
```

---

## ✅ Summary

### 🔹 Tokenization
Splits text into manageable units for NLP models.

---

### 🔹 Tokenizers
- NLTK  
- spaCy  
- Hugging Face  
- torchtext

---

### 🔹 Tokenization Types
- **Word-Based**: Semantically meaningful, large vocabulary  
- **Character-Based**: Smaller vocabulary, higher dimensionality  
- **Subword-Based**: Balanced — uses WordPiece, Unigram, SentencePiece

---

### 🔹 Libraries & Tools
- `transformers` (Hugging Face)  
- `torchtext` (PyTorch)

---

### 🔹 Practical Workflow

```text
Tokenize → Build Vocabulary → Convert to Indices → Add Special Tokens → Apply Padding
```

