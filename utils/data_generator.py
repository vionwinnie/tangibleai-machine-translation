from torch.utils.data import Dataset, DataLoader 
import pandas as pd
import numpy as np

# conver the data to tensors and pass to the Dataloader 
# to create an batch iterator

class MyData(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.target = y
        # TODO: convert this into torch code is possible
        self.length = [ np.sum(1 - np.equal(x, 0)) for x in X]
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        x_len = self.length[index]
        return x,y,x_len
    
    def __len__(self):
        return len(self.data)

# This class creates a word -> index mapping (e.g,. "dad" -> 5) and vice-versa 
# (e.g., 5 -> "dad") for each language,
class LanguageIndex():
    def __init__(self, phrases):
        """ `phrases` is a list of phrases in one language """
        self.word2idx = {}
        self.idx2word = {}  # this can just be a list
        self.vocab = set()
        self.create_index(phrases)
        
    def create_index(self, phrases):
        for phrase in phrases:
            # update with individual tokens
            self.vocab.update(phrase.split(' '))
            
        # sort the vocab
        self.vocab = sorted(self.vocab)

        # add a padding token with index 0
        self.word2idx['<pad>'] = 0
        
        # word to index mapping
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1 # +1 because of pad token
        
        # index to word mapping
        for word, index in self.word2idx.items():
            self.idx2word[index] = word   


if __name__=='__main__':
    phrases = ['<start> i am a man <end>','<start> weather is great <end>']
    test_language_index=LanguageIndex(phrases)
    print(test_language_index.word2idx)
    print(test_language_index.idx2word)

