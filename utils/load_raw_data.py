"""This module loads data"""

import pandas as pd
import tangiblemt.utils.preprocess as dp
from tangiblemt.utils.data_generator import MyData, LanguageIndex

def load_raw_text_file(file_path,num_examples=None):
    """
    Input: Path for raw data
    Output: Preprocessed Dataframe
    """
    lines = open(file_path, encoding='UTF-8').read().strip().split('\n')

    # creates lists containing each pair
    original_word_pairs = [[w for w in l.split('\t')] for l in lines]

    if num_examples:
        original_word_pairs = original_word_pairs[:num_examples]

    # Store data as a Pandas dataframe
    df = pd.DataFrame(original_word_pairs, columns=["eng", "es", 'info'])

    # Now we do the preprocessing using pandas and lambdas
    df["eng"] = df.eng.apply(lambda w: dp.preprocess_sentence(w))
    df["es"] = df.es.apply(lambda w: dp.preprocess_sentence(w))

    return df

def convert_tensor(df, inp_index, targ_index):

    """
    Convert sentences into tensors
    """

    # Vectorize the input and target languages
    input_tensor = [[inp_index.word2idx[s] for s in es.split(' ')]  for es in df["es"].values.tolist()]
    target_tensor = [[targ_index.word2idx[s] for s in eng.split(' ')]  for eng in df["eng"].values.tolist()]

    # calculate the max_length of input and output tensor
    max_length_inp, max_length_tar = dp.max_length(input_tensor), dp.max_length(target_tensor)

    # inplace padding
    input_tensor = [dp.pad_sequences(x, max_length_inp) for x in input_tensor]
    target_tensor = [dp.pad_sequences(x, max_length_tar) for x in target_tensor]

    return input_tensor, target_tensor
