import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Decoder(nn.Module):
    def __init__(self, config,vocab_size):
        super(Decoder, self).__init__()
        self.dec_units = config.get("decoder_hidden",64)
        self.enc_units = config.get("encoder_hidden",64)
        self.vocab_size = vocab_size
        self.embedding_dim = config.get("embedding_dim",256)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim,
                          self.dec_units,
                          batch_first=True)
        self.fc = nn.Linear(self.dec_units, self.vocab_size)
        self.attention=False

        # used for attention
        #self.W1 = nn.Linear(self.enc_units, self.dec_units)
        #self.W2 = nn.Linear(self.enc_units, self.dec_units)
        #self.V = nn.Linear(self.enc_units, 1)

    def forward(self, x, hidden, enc_output):
        # enc_output original: (max_length, batch_size, enc_units)
        # enc_output converted == (batch_size, max_length, hidden_size)
        enc_output = enc_output.permute(1,0,2)
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score

        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        if self.attention:
            hidden_with_time_axis = hidden.permute(1, 0, 2)
            # score: (batch_size, max_length, hidden_size) # Bahdanaus's
            # we get 1 at the last axis because we are applying tanh(FC(EO) + FC(H)) to self.V
            # It doesn't matter which FC we pick for each of the inputs
            score = torch.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))
            # attention_weights shape == (batch_size, max_length, 1)
            # we get 1 at the last axis because we are applying score to self.V
            attention_weights = torch.softmax(self.V(score), dim=1)

            # context_vector shape after sum == (batch_size, hidden_size)
            context_vector = attention_weights * enc_output
            context_vector = torch.sum(context_vector, dim=1)

            # x shape after passing through embedding == (batch_size, 1, embedding_dim)
            # takes case of the right portion of the model above (illustrated in red)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        #x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # ? Looks like attention vector in diagram of source
        if self.attention:
            x = torch.cat((context_vector.unsqueeze(1), x), -1)

        # passing the concatenated vector to the GRU
        # output: (batch_size, 1, hidden_size)
        output, state = self.gru(x)


        # output shape == (batch_size * 1, hidden_size)
        output =  output.view(-1, output.size(2))

        # output shape == (batch_size * 1, vocab)
        x = self.fc(output)
        
        if self.attention:
            return x, state, attention_weights
        else:
            return x, state

    def initialize_hidden_state(selfi,batch_size):
        return torch.zeros((1, batch_size, self.dec_units))

