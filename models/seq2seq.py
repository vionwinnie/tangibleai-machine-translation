import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from torch.autograd import Variable
from models.modules.encoders import Encoder
from models.modules.decoders import Decoder

class Seq2Seq(nn.Module):
    """
        Sequence to sequence module
    """

    def __init__(self, config,vocab_inp_size,vocab_out_size):
        super(Seq2Seq, self).__init__()
        self.SOS = config.get("start_index", 5)
        self.EOS = config.get("end_index",4)
        self.vocab_inp_size = vocab_inp_size 
        self.vocab_out_size = vocab_out_size
        self.batch_size = config.get("batch_size", 64)
        self.sampling_prob = config.get("sampling_prob", 0.)
        self.gpu = config.get("gpu", False)
        self.debug = config.get("debug",False)

        # Encoder
        self.encoder = Encoder(config,vocab_inp_size)

        # Decoder
        self.decoder = Decoder(config,vocab_out_size)

        # Loss Function
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

        print(config)

    ##TODO:
    def encode(self,x,x_len):
        """
        Given Input Sequence, Pass the Data to Encode

        Return:
        - Encoder Output
        - Encoder State
        """

        ## Check to see if batch_size parameter is fixed or base on input batch
        batch_size = x.size()[0]
        init_state = self.encoder.init_hidden(batch_size)
        encoder_outputs, encoder_state, input_lengths = self.encoder.forward(x, init_state, x_len)

        return encoder_outputs, encoder_state.squeeze(0)

    ##TODO:
    def decode(self, encoder_outputs, encoder_hidden, targets, targets_lengths, input_lengths):
        """
        Args:
            encoder_outputs: (B, T, H)
            encoder_hidden: (B, H)
            targets: (B, L)
            targets_lengths: (B)
            input_lengths: (B)
        Vars:
            decoder_input: (B)
            decoder_context: (B, H)
            hidden_state: (B, H)
            attention_weights: (B, T)
        Outputs:
            alignments: (L, T, B)
            logits: (B*L, V)
            labels: (B*L)
        """
        batch_size = encoder_outputs.size()[0]
        max_length = targets.size()[1]
        decoder_input = Variable(torch.LongTensor([self.SOS] * batch_size)).squeeze(-1)
        decoder_context = encoder_outputs.transpose(1, 0)[-1]
        decoder_hidden = encoder_hidden

        alignments = Variable(torch.zeros(max_length, encoder_outputs.size(1), batch_size))
        logits = Variable(torch.zeros(max_length, batch_size, self.decoder.output_size))

        if self.gpu:
            decoder_input = decoder_input.cuda()
            decoder_context = decoder_context.cuda()
            logits = logits.cuda()

        ## Generate output one time step at a time
        for t in range(max_length):

            # The decoder accepts, at each time step t :
            # - an input, [B]
            # - a context, [B, H]
            # - an hidden state, [B, H]
            # - encoder outputs, [B, T, H]


            # The decoder outputs, at each time step t :
            # - an output, [B]
            # - a context, [B, H]
            # - an hidden state, [B, H]
            # - weights, [B, T]

            outputs, hidden = self.decoder.forward(
                    input=decoder_input.long(),
                    hidden=decoder_hidden)

            logits[t] = outputs
            if self.training:
                decoder_input = targets[:, t]
            else:
                decoder_input = outputs ## TODO: Check if the dimensions are matching
        labels = targets.contiguous().view(-1)
            
        ## Return final state, labels and alignments
        return logits, labels.long(),alignments


