import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        all_words = [word for text in texts for word in text.lower().split()]
        vocab = set(all_words)

        for i, spl_tkn in enumerate(["<PAD>", "<UNK>", "<BOS>", "<EOS>"]):
            self.word_to_id[spl_tkn] = i
            self.id_to_word[i] = spl_tkn

        for j, word in enumerate(sorted(vocab)):
            idx = 4 + j
            self.word_to_id[word] =  idx
            self.id_to_word[idx] =  word

        self.vocab_size = len(self.word_to_id)
        
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        
        words = text.lower().split()
        return [self.word_to_id.get(word, self.word_to_id[self.unk_token]) for word in words]
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        
        return " ".join(self.id_to_word.get(id, self.unk_token) for id in ids)
