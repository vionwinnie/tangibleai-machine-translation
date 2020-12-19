"""This script contains functions used to convert between tokens and words for evaluation purpose"""
import numpy as np

def detokenize_sentences(token_array,token_dictionary,output='sentence'):
    sentences = np.vectorize(token_dictionary.get)(token_array)
    sentences.tolist()
    list_dummy_tokens = ['<start>','<end>','<pad>']

    reshaped_sentences = []

    for sentence in sentences:
        sentence_drop_tokens = [token for token in sentence if token not in list_dummy_tokens]
        if output=='sentence':
            concat_sentence = ' '.join(sentence_drop_tokens)
        else:
            concat_sentence = sentence_drop_tokens
        reshaped_sentences.append(concat_sentence)

    return reshaped_sentences

def count_bag_of_words(target_token_list,pred_token_list,
        output='sum',debug=False):
    
    """
    Input:
    - actual_token_list = List of List of token (int or string)
    - pred_token_list = List of List of token (int or string)
    
    Output:
    - pct_of_actual_token_contained = percentage of token in actual_token_list contained in pred_token_list
    """
    accum_accuracy = 0
    assert len(target_token_list) ==  len(pred_token_list)
    
    length_list = len(pred_token_list)
    
    for target_sent,pred_sent in zip(target_token_list,pred_token_list):
        ## Assume that the tokens are unique
        target_token_list = list(set(target_sent))
        pred_token_list = list(set(pred_sent))

        if debug:
            print("===============================")
            print("Target: {}".format(target_token_list))
            print("Pred : {}".format(pred_token_list))

        assert len(target_token_list) > 0

        num_contained = sum(actual in target_token_list for actual in pred_token_list)
        pct_contained = num_contained / len(target_token_list)
        
        accum_accuracy += pct_contained
        
    if output=='sum':
        return accum_accuracy
    else:
        ## Average accuracy
        return accum_accuracy/length_list
        

def convert_token_array_to_corpus(target,label,batch_size,idx2word_dict):

    decoded_targets = None
    decoded_labels = None

    if target:
        decoded_targets = detokenize_sentences(label.view(batch_size,-1),idx2word_dict)
    if label:
        decoded_labels = detokenize_sentences(target,idx2word_dict)
    
    return decoded_labels, decoded_targets


def bag_of_word_accuracy(actual_corpus,pred_corpus):
    
    assert len(actual_corpus) == len(pred_corpus)
    sum_accuracy = 0
    for actual_sentence,pred_sentence in zip(actual_corpus,pred_corpus):
        cur_accuracy = count_bag_of_words(actual_token_list,pred_token_list)
        sum_accuracy += cur_accuracy
    return sum_accuracy, sum_accuracy/len(actural_corpus)
