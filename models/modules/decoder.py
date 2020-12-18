""" This is the decoder module that decodes high-dimensional vector from Encoder"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Decoder(nn.Module):
    def __init__(self, config,vocab_size):
        super(Decoder, self).__init__()
        self.dec_units = config.get("decoder_hidden", 64)
        self.enc_units = config.get("encoder_hidden", 64)
        self.vocab_size = vocab_size
        self.embedding_dim = config.get("embedding_dim", 256)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim,
                          self.dec_units,
                          batch_first=True)
        self.fc = nn.Linear(self.dec_units, self.vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.attention = False
        self.debug=config.get("debug",False)
        # used for attention
        #self.W1 = nn.Linear(self.enc_units, self.dec_units)
        #self.W2 = nn.Linear(self.enc_units, self.dec_units)
        #self.V = nn.Linear(self.enc_units, 1)

    def forward(self, inputs, hidden):
        """
        INPUTS:
        - inputs: BATCHSIZE x 1 (Input Token)
        - hidden: Last Encoder Input 1x BATCHSIZE x Encoding Units
        """
        if self.debug:
            print("x: {}".format(inputs.shape))
            print("hidden: {}".format(hidden.shape))

        output = self.embedding(inputs)

        if self.debug:
            print("dimension x after embedding layer: {}".format(inputs.shape))
        
        output, state = self.gru(output,hidden)

        if self.debug:
            print("output dim:{}, state dim: {}".format(output.shape,state.shape))
        # output shape == (batch_size * 1, hidden_size)
        output = output.view(-1, output.size(2))
        if self.debug:
            print("output after reshape: {}".format(output.shape))

        # output shape == (batch_size * 1, vocab)
        output = self.fc(output)
        if self.debug:
            print("output after fully-connected: {}".format(output.shape))
      
        output = self.softmax(output)
        if self.debug:
            print("output after softmax: {}".format(output.shape))
        
        return output, state

    def initialize_hidden_state(self, batch_size):
        return torch.zeros((1, batch_size, self.dec_units))

