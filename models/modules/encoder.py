import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


 class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim, self.enc_units)
        
    def forward(self, x, lens, device=device):
        # x: batch_size, max_length 
        
        # x: batch_size, max_length, embedding_dim
        x = self.embedding(x) 
                
        # x transformed = max_len X batch_size X embedding_dim
        # x = x.permute(1,0,2)
        x = pack_padded_sequence(x, lens) # unpad
    
        self.hidden = self.initialize_hidden_state(device)
        
        # output: max_length, batch_size, enc_units
        # self.hidden: 1, batch_size, enc_units
        output, self.hidden = self.gru(x, self.hidden) # gru returns hidden state of all timesteps as well as hidden state at last timestep
        
        # pad the sequence to the max length in the batch
        output, _ = pad_packed_sequence(output)
        
        return output, self.hidden

    def initialize_hidden_state(self, device=device):
        return torch.zeros((1, self.batch_sz, self.enc_units)).to(device)


