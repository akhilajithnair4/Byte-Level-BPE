import regex as re
import os
import pickle

CORPUS_PATH = "C:\\Users\\L123110\\OneDrive - Eli Lilly and Company\\Desktop\\AI Practice\\tokenizer\\corpus.txt"
print(os.path.exists(CORPUS_PATH))


gpt2_split_pattern = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"


class BPETokenizer:

    def __init__(self):
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        
    def get_stats(self, tokenids):
        counts ={}
        pairs = [pair for chunks in tokenids for pair in zip(chunks,chunks[1:])]

        for pair in pairs:       
            counts[pair] = counts.get(pair,0) + 1

        return counts

    def merge_replace(self, ids, pair, idx):
        i =0 
        new_ids = []
        while i < len(ids):

            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1

        return new_ids
  
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
    

    def encode(self, text):
         tokens = re.findall(gpt2_split_pattern, text)
         token_ids = [list(token.encode('utf-8')) for token in tokens]

         while True:
            stats= self.get_stats(token_ids)

            if not stats:
                break
            pairs = min(stats, key= lambda p: self.merges.get(p, float('inf')))
            if pairs not in self.merges:
                break

            idx = self.merges[pairs]

            token_ids = [self.merge_replace(chunk, pairs, idx) for chunk in token_ids]   

         return [i for chunk in token_ids for i in chunk]
            
   
    def build_vocab(self):
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]


    def decode(self, ids):
        tokens =  [self.vocab[i] for i in ids]
        return b"".join(tokens).decode('utf-8', errors="replace")


    def save_merges(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.merges, f)

    def save_vocab(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.vocab, f)


    def load_merges(self, filepath):
        with open(filepath, "rb") as f:
            self.merges = pickle.load(f)

    def load_vocab(self, filepath):
        with open(filepath, "rb") as f:
            self.vocab = pickle.load(f)


if __name__ == "__main__":

    tokenizer2 = BPETokenizer()
    if os.path.exists("merges.pkl") and os.path.exists("vocab.pkl"):
         tokenizer2.load_merges("merges.pkl")     # fills tokenizer2.merges
         tokenizer2.load_vocab("vocab.pkl")       # fills tokenizer2.vocab
         print(f"Loaded {len(tokenizer2.merges)} merges")    # should print 1744

         token_ids = tokenizer2.encode("You are all resolved rather to die than to famish?")
         print(token_ids)                                    # [669]
         print(repr(tokenizer2.decode(token_ids)))           # ' time'

    else:
         #Reading the corpus file
        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            text = f.read()
        tokenizer2.train(text, vocab_size=2000)
        tokenizer2.save_merges("merges.pkl")
        tokenizer2.save_vocab("vocab.pkl")
    
   