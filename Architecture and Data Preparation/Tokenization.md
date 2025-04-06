# ✂️ Tokenization in Natural Language Processing (NLP)

Tokenization is a fundamental step in processing natural language for AI models. It involves breaking down sentences into smaller units, called **tokens**, which can be words, subwords, or characters. These tokens help models better understand and analyze text data.

---

## 📌 What is Tokenization?

- Tokenization is the process of breaking text into smaller parts (tokens) for model input.
- Example:  
  `"IBM taught me tokenization"` → `["IBM", "taught", "me", "tokenization"]`
- Tokens can vary depending on the model and tokenizer used.

---

## ⚙️ Tokenizers

A **tokenizer** is a program that splits text into tokens. It can operate in multiple ways depending on the type of tokenization method applied.

### Three Main Tokenization Methods:

---

### 1. **Word-Based Tokenization**
- Splits text into words.
- **Pros**: Preserves semantic meaning.
- **Cons**: Large vocabulary size.
- Tools like **NLTK** and **spaCy** apply this method.
- Limitation: Similar words (e.g., `unicorn`, `unicorns`) are treated differently.

---

### 2. **Character-Based Tokenization**
- Splits text into individual characters.
- **Pros**: Small vocabulary.
- **Cons**: Higher dimensionality, lower semantic value.
- Example:  
  `"This is a sentence"` → `["T", "h", "i", "s", " ", "i", "s", ...]`

---

### 3. **Subword-Based Tokenization**
- Frequently used words remain whole.
- Rare words are split into subwords.
- **Pros**: Balances vocab size and semantic integrity.

#### Subword Tokenization Algorithms:
- **WordPiece**: Merges or splits symbols based on value.
- **Unigram**: Starts with many token options, narrows down by frequency.
- **SentencePiece**: Segments text and assigns unique IDs.

---

## 🧪 Tokenization with Transformers (Hugging Face)

### Example: WordPiece with BERT
```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize("Tokenization using WordPiece")
