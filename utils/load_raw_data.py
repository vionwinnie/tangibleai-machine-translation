import pandas as pd
import tangiblemt.utils.preprocess as dp


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
