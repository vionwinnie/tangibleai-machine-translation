""" This is the encoder module encapsulated within Seq2Seq """
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, config,vocab_size):
        super(Encoder, self).__init__()
        self.gpu =  config.get('gpu',False)
        self.enc_units = config.get('encoder_hidden',1024)
        self.vocab_size = vocab_size
        self.embedding_dim = config.get('embedding_dim',256)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.device = torch.device("cuda" if self.gpu else "cpu")
        self.lstm = nn.GRU(self.embedding_dim, self.enc_units)
        self.debug = config.get('debug',False)
        self.hidden = None

    def forward(self, x, init_state,lens):
        # x: batch_size, max_length 
        if self.debug:
            print("Input size: {}".format(x.shape))
        # x: batch_size, max_length, embedding_dim
        x = self.embedding(x) 
        if self.debug:
            print("After embedding layer: {}".format(x.shape))
        # x transformed = max_len X batch_size X embedding_dim
        # x = x.permute(1,0,2)
        x = pack_padded_sequence(x, lens)
    
        self.hidden = init_state
        # output: max_length, batch_size, enc_units
        # self.hidden: 1, batch_size, enc_units
        output, self.hidden = self.lstm(x, self.hidden) 
        # LSTM returns hidden state of all timesteps as well as hidden state at last timestep (which is the output) in PackedSequence data format

        # unpad the sequence to the max length in the batch (max_length,batch_size,num_encoding_units)
        output, _ = pad_packed_sequence(output)
        
        if self.debug:
            print('output after unpack : {}'.format(output.shape))
        
        return output, self.hidden

    def initialize_hidden_state(self,batch_size):
        output = torch.zeros((1, batch_size, self.enc_units)).to(self.device)
        if self.debug:
            print("initialized hidden state dim : {}".format(output.shape))
        return output


