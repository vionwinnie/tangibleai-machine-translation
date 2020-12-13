import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, config,vocab_size):
        super(Encoder, self).__init__()
        self.gpu =  config.get('gpu',False)
        self.batch_sz = config.get('batch_size',64)
        self.enc_units = config.get('encoder_hidden',1024)
        self.vocab_size = vocab_size
        self.embedding_dim = config.get('embedding_dim',256)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.device = torch.device("cuda" if self.gpu else "cpu")
        self.gru = nn.GRU(self.embedding_dim, self.enc_units)
        self.debug = config.get('debug',False)

    def forward(self, x, lens):
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
    
        self.hidden = self.initialize_hidden_state()
        
        # output: max_length, batch_size, enc_units
        # self.hidden: 1, batch_size, enc_units
        output, self.hidden = self.gru(x, self.hidden) 
        # gru returns hidden state of all timesteps as well as hidden state at last timestep (which is the output) in PackedSequence data format

        # unpad the sequence to the max length in the batch
        output, _ = pad_packed_sequence(output)
        
        if self.debug:
            print('output after unpack : {}'.format(output.shape))
        
        return output, self.hidden

    def initialize_hidden_state(self):
        output = torch.zeros((1,self.batch_sz,self.enc_units)).to(self.device)
        if self.debug:
            print("initialized hidden state dim : {}".format(output.shape))
        return output


