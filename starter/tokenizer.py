"""Baseline tokenizer: raw UTF-8 bytes, vocab of 256. Simple, never fails on
unseen text — and treats a Devanagari character as 3 tokens. Think about
what that does to your model's context window and your token budget on the
Hindi part of the corpus.

You may replace this with anything you train ON THE PROVIDED CORPUS ONLY
(e.g., BPE), as long as:
  1. it can encode ARBITRARY UTF-8 text (byte-level fallback) and it is
     LOSSLESS: decode(encode(text)) == text, exactly. The scorer and the
     graders both verify this round-trip — a lossy tokenizer makes bpb
     meaningless and disqualifies the run.
  2. this file keeps exposing:  load() -> tokenizer object with
     .encode(str) -> list[int], .decode(list[int]) -> str, .vocab_size.
     train.py and evaluate.py call load() with NO arguments — keep any
     extra parameters optional.
  3. anything it needs is saved under your submission folder and loaded by
     load() with no internet. Grading runs with cwd = your folder; resolve
     saved files relative to __file__ to be safe.
"""
"""Byte-Pair Encoding Tokenizer trained on the corpus."""
import json
import os

class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.merges = {} 
        self.vocab = {i: bytes([i]) for i in range(256)}
        
    def train(self, text):
        tokens = list(text.encode("utf-8"))
        num_merges = self.vocab_size - 256
        
        for i in range(num_merges):
            stats = {}
            for pair in zip(tokens, tokens[1:]):
                stats[pair] = stats.get(pair, 0) + 1
            
            if not stats:
                break
            
            best = max(stats, key=stats.get)
            if stats[best] < 2:
                break 
            
            new_id = 256 + i
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            
            # Apply merge to token sequence
            new_tokens = []
            j = 0
            while j < len(tokens):
                if j < len(tokens) - 1 and (tokens[j], tokens[j+1]) == best:
                    new_tokens.append(new_id)
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1
            tokens = new_tokens

        self.vocab_size = 256 + len(self.merges)

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                if pair in self.merges:
                    stats[pair] = self.merges[pair]
            
            if not stats:
                break 
                
            best = min(stats, key=stats.get)
            
            new_tokens = []
            j = 0
            while j < len(tokens):
                if j < len(tokens) - 1 and (tokens[j], tokens[j+1]) == best:
                    new_tokens.append(self.merges[best])
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1
            tokens = new_tokens
        return tokens

    def decode(self, ids):
        b = bytearray()
        for idx in ids:
            if idx in self.vocab:
                b.extend(self.vocab[idx])
            else:
                b.extend(bytes([idx % 256])) # Byte-level fallback (Required)
        return b.decode("utf-8", errors="replace")

    def save(self, path):
        merges_str = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        with open(path, "w") as f:
            json.dump({"vocab_size": self.vocab_size, "merges": merges_str}, f)
            
    def load_weights(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]
        self.merges = {}
        for k, v in data["merges"].items():
            p1, p2 = map(int, k.split(","))
            self.merges[(p1, p2)] = v
            
        self.vocab = {i: bytes([i]) for i in range(256)}
        for pair, idx in self.merges.items():
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

def load(path=None):
    tok = BPETokenizer(vocab_size=1000)
    bpe_path = os.path.join(os.path.dirname(__file__), "bpe_vocab.json")
    
    if os.path.exists(bpe_path):
        tok.load_weights(bpe_path)
    else:
        # Train on the fly for the first time
        data_path = os.path.join(os.path.dirname(__file__), "../data/train_corpus.txt")
        print(f"Training BPE tokenizer (vocab_size={tok.vocab_size}). Please wait ~30-60 seconds...")
        text = open(data_path, encoding="utf-8").read()
        tok.train(text)
        tok.save(bpe_path)
        print("BPE Tokenizer trained and saved to bpe_vocab.json!")
    return tok