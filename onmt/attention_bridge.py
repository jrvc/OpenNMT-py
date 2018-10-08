"""Multi-headed attention"""

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

#from torch.nn.utils.rnn import pack_padded_sequence as pack
#from torch.nn.utils.rnn import pad_packed_sequence as unpack

#from onmt.encoders.encoder import EncoderBase
#from onmt.utils.rnn_factory import *
from onmt.decoders.decoder import RNNDecoderState
#from onmt.decoders import decoder

class AttentionBridge(nn.Module):
    """
    Multi-headed attention. Bridge between encoders->decoders
    """
    def __init__(self, hidden_size, attention_heads, dec_num_layers, dropout=0.05):
        """Attention Heads Layer:"""
        super(AttentionBridge, self).__init__()
        u = hidden_size
        d = hidden_size
        r = attention_heads
        self.drop = nn.Dropout(dropout)
        self.ws1 = nn.Linear(d, u, bias=False)
        self.ws2 = nn.Linear(d, r, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        #        self.init_weights()
        self.attention_hops = r
        self.hidden_size = hidden_size
        self.M = None
        self.dec_layers = dec_num_layers
        '''Decoder initialization: '''
        #u = self.hidden_dim
        #r = attentionhops
        #self.attention_hops = r
        self.W_init = nn.Linear(u,dec_num_layers*u , bias=False)
        self.tanh = nn.Tanh()


    def forward(self, enc_output):
        #import ipdb; ipdb.set_trace(context=5)
        output, alphas = self.mixAtt(enc_output)
        #take transpose to match dimensions s.t. r=new_seq_len:
        self.M = torch.transpose(output, 0, 1).contiguous() #[r,bsz,nhid]
        #import ipdb; ipdb.set_trace(context=10)
        h_avrg = (self.M).mean(dim=0, keepdim=True)

        return h_avrg, self.M # enc_final=h_avrg memory_bank=output3



    def mixAtt(self, outp):
        """Notation based on Lin et al. (2017) A structured self-attentive sentence embedding"""
        outp = torch.transpose(outp, 0, 1).contiguous()
        size = outp.size()  # [bsz, len, nhid]
        compressed_embeddings = outp.view(-1, size[2])  # [bsz*len, nhid*2]
        hbar = self.tanh(self.ws1(self.drop(compressed_embeddings)))  # [bsz*len, attention-unit]
        alphas = self.ws2(hbar).view(size[0], size[1], -1)  # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]
        alphas = alphas.view(-1, size[1]) # [bsz*hop, len]
        alphas = self.softmax(alphas)  # [bsz*hop, len]
        alphas = alphas.view(size[0], self.attention_hops, size[1])  # [bsz, hop, len]
        return torch.bmm(alphas, outp), alphas

    def init_decoder_state(self, src, memory_bank, encoder_final):
        '''
        initialize the decoder state, `s_0`.
        We use a similar s_0 as in rSennrich et al.(2017) "Nematus":
             - s_0 = tanh(W_init * h_avrg);
                  where h_avrg is the average of the sentence embedding, M,
                  (instead of taking the average over the hidden states of
                   the RNN, as in rSennrich(2017) over the )
        '''
        #import ipdb; ipdb.set_trace()
        s_0 = self.tanh(self.W_init(encoder_final))
        ss = s_0[:,:,:self.hidden_size]
        for i in range(self.dec_layers - 1):
            init = (i+1) * self.hidden_size
            end = (i+2) * self.hidden_size
            ss = torch.cat([ss,s_0[:,:,init:end]])
        s_0 = ss
        return RNNDecoderState(self.hidden_size, s_0)
