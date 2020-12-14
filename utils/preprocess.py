import unicodedata,re
import pandas as pd
import numpy as np

# Converts the unicode file to ascii
def unicode_to_ascii(s):
    """
    Normalizes latin chars with accent to their canonical decomposition
    """
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ." 
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    
    w = w.rstrip().strip()
    
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w

def max_length(tensor):
    return max(len(t) for t in tensor)

def pad_sequences(x, max_len):
    padded = np.zeros((max_len), dtype=np.int64)
    if len(x) > max_len: padded[:] = x[:max_len]
    else: padded[:len(x)] = x
    return padded

### sort batch function to be able to use with pad_packed_sequence
### Can be use for bucketing later to reduce time complexity while training
def sort_batch(X, y, x_length,y_length):
    x_length, indx = x_length.sort(dim=0, descending=True)
    X = X[indx]
    y = y[indx]
    y_length = y_length[indx]
    return X.transpose(0,1), y, x_length, y_length # transpose (batch x seq) to (seq x batch)

if __name__=='__main__':
    sentence = "I am a man?"
    print(preprocess_sentence(sentence))

    phrase = [2,3,4,5]
    print(pad_sequences(phrase,max_len=16))
