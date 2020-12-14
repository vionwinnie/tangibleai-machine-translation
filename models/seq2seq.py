import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from torch.autograd import Variable

from tangiblemt.utils.preprocess import sort_batch
from tangiblemt.models.modules.encoder import Encoder
from tangiblemt.models.modules.decoder import Decoder
from tangiblemt.models.helpers import mask_3d

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
        self.gpu = config.get("gpu", False)
        self.debug = config.get("debug",False)
        self.training = False
        
        if config.get('loss') == 'cross_entropy':
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

        # Encoder
        self.encoder = Encoder(config,vocab_inp_size)

        # Decoder
        self.decoder = Decoder(config,vocab_out_size)

        # Loss Function
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

        print(config)

    def encode(self,x,x_len):
        """
        Given Input Sequence, Pass the Data to Encode

        Return:
        - Encoder Output
        - Encoder State
        """

        ## Check to see if batch_size parameter is fixed or base on input batch
        cur_batch_size = x.size()[1]
        encode_init_state = self.encoder.initialize_hidden_state(cur_batch_size)
        encoder_outputs, encoder_state = self.encoder.forward(x, encode_init_state, x_len)

        return encoder_outputs, encoder_state

    def decode(self, encoder_outputs, encoder_hidden, targets, targets_lengths):
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

        ## TODO: Should this be a variable?
        max_length = targets.size()[1]
        decoder_input = Variable(torch.LongTensor([self.SOS] * batch_size)).squeeze(-1)
        decoder_hidden = encoder_hidden
    
        logits = Variable(torch.zeros(max_length, batch_size, self.decoder.vocab_size))
        
        if self.gpu:
            decoder_input = decoder_input.cuda()
            logits = logits.cuda()
        
        for t in range(1,out_batch.size(1)):
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
            
            # enc_hidden: 1, batch_size, enc_units
            # output: max_length, batch_size, enc_units
            predictions, decoder_hidden = self.decoder.forward(decoder_input.to(device),
                                         decoder_hidden.to(device),
                                         encoder_outputs.to(device))

            ## Store Prediction at time step t
            logits[t] = predictions

            if self.training:
                decoder_input = out_batch[:, t].unsqueeze(1)
            else:
                decoder_input = torch.argmax(predictions,axis=1).unsqueeze(1)

        labels = targets.contiguous().view(-1)
        mask_value = 0

        ## Masking the logits to prepare for eval
        logits = mask_3d(logits.transpose(1,0),targets_lengths,mask_value)
        logits = logits.contiguous().view(-1,self.output_vocab_size)
        if debug:
            print("Logit dimension: {}, labels dimension".format(logits.shape,labels.shape))

        ## Return final state, labels 
        return logits, labels.long()

    def step(self, batch):
        x, y, x_len,y_len = batch

        ## sort the batch for pack_padded_seq in forward function
        x_sorted, y_sorted, x_len_sorted, y_len_sorted = sort_batch(x,y,x_len,y_len)
        if self.debug:
            print("x_sorted: {}".format(x_sorted.shape))
            print("y_sorted: {}".format(y_sorted.shape))
            print("x_len_sorted: {}".format(x_len_sorted.shape))
            print("y_len_sorted: {}".format(y_len_sorted.shape))
        if self.gpu:
            x_sorted = x_sorted.cuda()
            y_sorted = y_sorted.cuda()
            x_len_sorted = x_len_sorted.cuda()
            y_len_sorted = y_len_sorted.cuda()
        
        encoder_out, encoder_state = self.encode(x_sorted, x_len_sorted)
        logits, labels = self.decode(encoder_out, encoder_state, y_sorted, y_len_sorted)
        return logits, labels

    def loss(self, batch):
        logits, labels = self.step(batch)
        loss = self.loss_fn(logits, labels)
        return loss, logits, labels
