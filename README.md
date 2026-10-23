# Byte-Level BPE Tokenizer - Complete Beginner's Guide

## What is BPE (Byte Pair Encoding)?

BPE is a compression technique that breaks text into smaller pieces called **tokens**. Think of it like this:

**Without BPE:**
- Text: "hello world" 
- Every character = 1 token: `['h','e','l','l','o',' ','w','o','r','l','d']` = 11 tokens

**With BPE:**
- Text: "hello world"
- Common pairs merged: `['he', 'llo', ' wo', 'rld']` = 4 tokens (compressed!)

---

## Code Breakdown Step-by-Step

### 1. **Import and Setup**

```python
import regex as re
import os
import pickle

gpt2_split_pattern = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
```

- **regex**: A powerful pattern-matching library (more features than built-in `re`)
- **pickle**: Saves/loads Python objects to files (for vocab and merges)
- **gpt2_split_pattern**: A regex pattern that splits text into tokens intelligently
  - Example: `"hello world"` → `['hello', ' world']` (keeps words intact)

---

### 2. **Class Initialization**

```python
def __init__(self):
    self.vocab = {i: bytes([i]) for i in range(256)}
    self.merges = {}
```

**What it does:**
- `self.vocab`: Dictionary mapping token IDs to byte sequences
  - Starts with IDs 0-255 (all single bytes)
  - ID 0 = byte `\x00`, ID 65 = byte `A`, etc.
  - After training, IDs 256+ represent merged pairs
- `self.merges`: Dictionary storing which pairs were merged
  - Example: `{(104, 101): 258}` means bytes (h, e) = token 258

**Example:**
```python
vocab[65]     # bytes([65]) = b'A'
vocab[256]    # After training: b'he' (merged pair)
```

---

### 3. **get_stats() - Find Most Frequent Pairs**

```python
def get_stats(self, tokenids):
    counts = {}
    pairs = [pair for chunks in tokenids for pair in zip(chunks,chunks[1:])]
    
    for pair in pairs:       
        counts[pair] = counts.get(pair, 0) + 1
    
    return counts
```

**What it does:**
- Takes all token sequences and finds adjacent byte pairs
- Counts how many times each pair appears

**Example:**
```python
# Input: tokenids = [[104, 101, 108, 108, 111]]  # "hello"
# Pairs: (104,101), (101,108), (108,108), (108,111)
# Output: {
#   (104, 101): 1,  # "he"
#   (101, 108): 1,  # "el"
#   (108, 108): 1,  # "ll"
#   (108, 111): 1   # "lo"
# }

# If you had TWO "hello"s:
# tokenids = [[104,101,108,108,111], [104,101,108,108,111]]
# (104, 101) appears 2 times! Most frequent!
```

---

### 4. **merge_replace() - Merge a Pair**

```python
def merge_replace(self, ids, pair, idx):
    i = 0 
    new_ids = []
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids
```

**What it does:**
- Replaces all occurrences of a pair with a new token ID

**Example:**
```python
# Merge (32, 116) to token 257  [space + 't']
# Before: [104, 101, 32, 116, 101, 114]  # "he ter"
#                       ↓↓
# After:  [104, 101, 257, 101, 114]       # "he 257er"
```

---

### 5. **train() - Build BPE Vocabulary**

```python
def train(self, text, vocab_size):
    tokens = re.findall(gpt2_split_pattern, text[:700000])
    token_ids = [list(token.encode('utf-8')) for token in tokens]
    
    idx = 256
    merge_rounds = vocab_size - idx
    for _ in range(merge_rounds):
        frequent_pair = self.get_stats(token_ids)
        if not frequent_pair:
            break
        
        pair = max(frequent_pair, key=frequent_pair.get)
        print(f"Merging pair: {pair} -> {idx}")
        token_ids = [self.merge_replace(chunk, pair, idx) for chunk in token_ids]
        
        self.merges[pair] = idx
        idx += 1
    self.build_vocab()
```

**Step-by-step:**

1. **Regex split**: Convert text to tokens
   ```python
   text = "Hello world"
   tokens = ["Hello", " world"]
   ```

2. **UTF-8 encode**: Convert each token to bytes
   ```python
   token_ids = [
       [72, 101, 108, 108, 111],      # "Hello"
       [32, 119, 111, 114, 108, 100]  # " world"
   ]
   ```

3. **Merge loop**: Repeat until we have `vocab_size` tokens
   - Find most frequent pair across all tokens
   - Replace with new ID
   - Store the merge rule
   
   **Example iterations:**
   ```
   Iteration 0: Merge (101, 108) -> 256  # "el" is most common
   Iteration 1: Merge (108, 108) -> 257  # "ll" is most common
   Iteration 2: Merge (256, 257) -> 258  # merged pair is now most common
   ...
   ```

---

### 6. **build_vocab() - Create Token to Bytes Mapping**

```python
def build_vocab(self):
    for (p0, p1), idx in self.merges.items():
        self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
```

**What it does:**
- For each merge, combine the bytes of the two merged tokens

**Example:**
```python
# After training:
self.merges = {(104, 101): 256}  # merged h+e

# build_vocab creates:
self.vocab[256] = self.vocab[104] + self.vocab[101]
self.vocab[256] = b'h' + b'e' = b'he'
```

---

### 7. **encode() - Tokenize New Text**

```python
def encode(self, text):
    tokens = re.findall(gpt2_split_pattern, text)
    token_ids = [list(token.encode('utf-8')) for token in tokens]
    
    while True:
        stats = self.get_stats(token_ids)
        if not stats:
            break
        
        pairs = min(stats, key=lambda p: self.merges.get(p, float('inf')))
        if pairs not in self.merges:
            break
        
        idx = self.merges[pairs]
        token_ids = [self.merge_replace(chunk, pairs, idx) for chunk in token_ids]
    
    return [i for chunk in token_ids for i in chunk]
```

**What it does:**
- Takes new text and applies all learned merges to compress it

**Example:**
```python
# Input: "hello"
# Step 1: Encode to bytes: [104, 101, 108, 108, 111]
# Step 2: Apply merges (if "he" was learned):
#         Find pair (104, 101) in merges
#         Replace: [256, 108, 108, 111]  # 256 = "he"
# Step 3: Continue applying merges until none left
# Output: [256, 257, 258]  # Highly compressed!
```

---

### 8. **decode() - Convert Tokens Back to Text**

```python
def decode(self, ids):
    tokens = [self.vocab[i] for i in ids]
    return b"".join(tokens).decode('utf-8', errors="replace")
```

**What it does:**
- Takes token IDs and converts them back to text

**Example:**
```python
# Input: [256, 257]
# Step 1: Look up vocab
#         vocab[256] = b'he'
#         vocab[257] = b'llo'
# Step 2: Join: b'hello'
# Step 3: Decode to string: "hello"
```

---

### 9. **save/load Methods - Persist Training**

```python
def save_merges(self, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(self.merges, f)

def load_merges(self, filepath):
    with open(filepath, "rb") as f:
        self.merges = pickle.load(f)
```

**What it does:**
- `pickle.dump`: Saves Python object to file
- `pickle.load`: Loads Python object from file
- Allows reusing trained vocabulary without retraining

---

### 10. **Main Program Flow**

```python
if __name__ == "__main__":
    tokenizer2 = BPETokenizer()
    
    if os.path.exists("merges.pkl") and os.path.exists("vocab.pkl"):
        # Already trained - load and use
        tokenizer2.load_merges("merges.pkl")
        tokenizer2.load_vocab("vocab.pkl")
        
        token_ids = tokenizer2.encode("Hello world")
        print(tokenizer2.decode(token_ids))
    
    else:
        # First time - train from scratch
        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            text = f.read()
        
        tokenizer2.train(text, vocab_size=2000)
        tokenizer2.save_merges("merges.pkl")
        tokenizer2.save_vocab("vocab.pkl")
```

**Flow:**
1. Check if vocab already exists
2. If yes: Load and use for encoding
3. If no: Train on corpus, save for future use

---

## Complete Example

```python
# Initialize
tokenizer = BPETokenizer()

# TRAINING (first time only)
text = "the cat sat on the mat"
tokenizer.train(text, vocab_size=300)
tokenizer.save_merges("merges.pkl")
tokenizer.save_vocab("vocab.pkl")

# ENCODING (compress text)
token_ids = tokenizer.encode("the cat")
# Result: [256, 257, 258]  (highly compressed!)

# DECODING (decompress back to text)
decoded = tokenizer.decode(token_ids)
# Result: "the cat"
```

---

## Key Concepts Summary

| Concept | Explanation | Example |
|---------|-------------|---------|
| **Byte** | Single value 0-255 | 104 = 'h', 101 = 'e' |
| **Pair** | Two consecutive bytes | (104, 101) = "he" |
| **Token ID** | Number representing bytes/pairs | 256, 257, 258 |
| **Vocab** | Mapping of IDs to actual bytes | {256: b'he', 257: b'llo'} |
| **Merge** | Combining frequent pairs into tokens | (h, e) → "he" = token 256 |
| **Encode** | Convert text to token IDs | "hello" → [256, 257] |
| **Decode** | Convert token IDs back to text | [256, 257] → "hello" |

---

## Why BPE Matters

- **Compression**: 1 million characters → 100k tokens (10x reduction!)
- **Efficiency**: AI models process fewer tokens = faster inference
- **Balance**: Doesn't require full dictionary; learns patterns from data
- **Language agnostic**: Works with any text

This is how GPT, Claude, and other large language models tokenize text!
