"""Audio encoder"""
import math

import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from onmt.utils.rnn_factory import rnn_factory
from onmt.utils.cnn_factory import shape_transform, StackedCNN
from onmt.encoders.encoder import EncoderBase


class AudioEncoder(EncoderBase):
    """A simple encoder CNN -> RNN for audio input.

    Args:
        rnn_type (str): Type of RNN (e.g. GRU, LSTM, etc).
        enc_layers (int): Number of encoder layers.
        dec_layers (int): Number of decoder layers.
        brnn (bool): Bidirectional encoder.
        enc_rnn_size (int): Size of hidden states of the rnn.
        dec_rnn_size (int): Size of the decoder hidden states.
        enc_pooling (str): A comma separated list either of length 1
            or of length ``enc_layers`` specifying the pooling amount.
        dropout (float): dropout probablity.
        sample_rate (float): input spec
        window_size (int): input spec
    """

    def __init__(self, rnn_type, enc_layers, dec_layers, brnn,
                 enc_rnn_size, dec_rnn_size, enc_pooling, dropout,
                 sample_rate, window_size, enc_input_size):
        super(AudioEncoder, self).__init__()
        self.enc_layers = enc_layers
        self.rnn_type = rnn_type
        self.dec_layers = dec_layers
        num_directions = 2 if brnn else 1
        self.num_directions = num_directions
        assert enc_rnn_size % num_directions == 0
        enc_rnn_size_real = enc_rnn_size // num_directions
        assert dec_rnn_size % num_directions == 0
        self.dec_rnn_size = dec_rnn_size
        dec_rnn_size_real = dec_rnn_size // num_directions
        self.dec_rnn_size_real = dec_rnn_size_real
        self.dec_rnn_size = dec_rnn_size
        input_size = enc_input_size
        enc_pooling = enc_pooling.split(',')
        assert len(enc_pooling) == enc_layers or len(enc_pooling) == 1
        if len(enc_pooling) == 1:
            enc_pooling = enc_pooling * enc_layers
        enc_pooling = [int(p) for p in enc_pooling]
        self.enc_pooling = enc_pooling

        if type(dropout) is not list:
            dropout = [dropout]
        if max(dropout) > 0:
            self.dropout = nn.Dropout(dropout[0])
        else:
            self.dropout = None
        self.W = nn.Linear(enc_rnn_size, dec_rnn_size, bias=False)
        self.batchnorm_0 = nn.BatchNorm1d(enc_rnn_size, affine=True)
        self.rnn_0, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=input_size,
                        hidden_size=enc_rnn_size_real,
                        num_layers=1,
                        dropout=dropout[0],
                        bidirectional=brnn)
        self.pool_0 = nn.MaxPool1d(enc_pooling[0])
        for l in range(enc_layers - 1):
            batchnorm = nn.BatchNorm1d(enc_rnn_size, affine=True)
            rnn, _ = \
                rnn_factory(rnn_type,
                            input_size=enc_rnn_size,
                            hidden_size=enc_rnn_size_real,
                            num_layers=1,
                            dropout=dropout[0],
                            bidirectional=brnn)
            setattr(self, 'rnn_%d' % (l + 1), rnn)
            setattr(self, 'pool_%d' % (l + 1),
                    nn.MaxPool1d(enc_pooling[l + 1]))
            setattr(self, 'batchnorm_%d' % (l + 1), batchnorm)

    @classmethod
    def from_opt(cls, opt, embeddings=None):
        """Alternate constructor."""
        if embeddings is not None:
            raise ValueError("Cannot use embeddings with AudioEncoder.")
        return cls(
            opt.rnn_type,
            opt.enc_layers,
            opt.dec_layers,
            opt.brnn,
            opt.enc_rnn_size,
            opt.dec_rnn_size,
            opt.audio_enc_pooling,
            opt.dropout,
            opt.sample_rate,
            opt.window_size,
            opt.n_mels*opt.n_stacked_mels)

    def forward(self, src, lengths=None):
        """See :func:`onmt.encoders.encoder.EncoderBase.forward()`"""
        batch_size, _, nfft, t = src.size()
        src = src.transpose(0, 1).transpose(0, 3).contiguous() \
                 .view(t, batch_size, nfft)
        orig_lengths = lengths
        lengths = lengths.view(-1).tolist()

        assert nfft == self.rnn_0.input_size, " \n\t Verify options -n_mels [default=80] and -n_stacked_mels [default=1]. \n\t Must have the same value as specified during preprocessing. \n\t The AudioEncoder takes inputs of size n_mels*n_stacked_mels"
        for l in range(self.enc_layers):
            rnn = getattr(self, 'rnn_%d' % l)
            pool = getattr(self, 'pool_%d' % l)
            batchnorm = getattr(self, 'batchnorm_%d' % l)
            stride = self.enc_pooling[l]
            packed_emb = pack(src, lengths)
            memory_bank, tmp = rnn(packed_emb)
            memory_bank = unpack(memory_bank)[0]
            t, _, _ = memory_bank.size()
            memory_bank = memory_bank.transpose(0, 2)
            memory_bank = pool(memory_bank)
            lengths = [int(math.floor((length - stride) / stride + 1))
                       for length in lengths]
            memory_bank = memory_bank.transpose(0, 2)
            src = memory_bank
            t, _, num_feat = src.size()
            src = batchnorm(src.contiguous().view(-1, num_feat))
            src = src.view(t, -1, num_feat)
            if self.dropout and l + 1 != self.enc_layers:
                src = self.dropout(src)

        memory_bank = memory_bank.contiguous().view(-1, memory_bank.size(2))
        memory_bank = self.W(memory_bank).view(-1, batch_size,
                                               self.dec_rnn_size)

        state = memory_bank.new_full((self.dec_layers * self.num_directions,
                                      batch_size, self.dec_rnn_size_real), 0)
        if self.rnn_type == 'LSTM':
            # The encoder hidden is  (layers*directions) x batch x dim.
            encoder_final = (state, state)
        else:
            encoder_final = state
        return encoder_final, memory_bank, orig_lengths.new_tensor(lengths)

    def update_dropout(self, dropout):
        self.dropout.p = dropout
        for i in range(self.enc_layers - 1):
            getattr(self, 'rnn_%d' % i).dropout = dropout





from onmt.modules import MultiHeadedAttention
from onmt.encoders.transformer import TransformerEncoderLayer

class AudioEncoderTrf(EncoderBase):
    """A 2xCNN -> LxTrf encoder for audio input.

    Args:
        enc_layers (int): Number of encoder layers.
        hidden_size (int): Size of hidden states of the rnn.
        enc_pooling (str): A comma separated list either of length 1
            or of length ``enc_layers`` specifying the pooling amount.
        dropout (float): dropout probablity.
        window_size (int): input spec
    """
    def __init__(self, enc_layers, hidden_size, 
                 dropout, cnn_kernel_width, 
                 enc_input_size, heads, transformer_ff, max_relative_positions):
        super(AudioEncoderTrf, self).__init__()
        
        # cnn part of the encoder:
        self.enc_layers = enc_layers
        self.input_size = enc_input_size
        self.cnn_kernel_width = cnn_kernel_width
        self.cnn = StackedCNN(num_layers=2, input_size=hidden_size,
                              cnn_kernel_width=cnn_kernel_width, dropout=dropout)
        
        self.linear = nn.Linear(self.input_size, hidden_size)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        
        #self.batchnorm_0 = nn.BatchNorm1d(enc_rnn_size, affine=True) # is this needed?
        
        # trf part of the encoder:
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(
                d_model=hidden_size, heads=heads, d_ff=transformer_ff, dropout=dropout,
                max_relative_positions=max_relative_positions)
             for i in range(enc_layers)])
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)


    @classmethod
    def from_opt(cls, opt, embeddings=None):
        """Alternate constructor."""
        if embeddings is not None:
            raise ValueError("Cannot use embeddings with AudioEncoderTrf.")
        return cls(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.cnn_kernel_width,
            opt.n_mels*opt.n_stacked_mels,
            opt.heads,
            opt.transformer_ff,
            opt.max_relative_positions)
    


    def forward(self, src, lengths=None):
        """See :func:`onmt.encoders.encoder.EncoderBase.forward()`"""
        batch_size, _, input_dim, src_len = src.size() #[bsz,1,input_hsz,src_len]
        orig_lengths = lengths
        lengths = lengths.view(-1).tolist() #[bsz,input_hsz,src_len,1]
        
        # correct shape for FWDnn                                   
        src = src.transpose(2,3).transpose(0,1).contiguous().view(batch_size, src_len, input_dim) #[bsz, src_len, input_hsz]
        src_reshape = src.view(src.size(0) * src.size(1), -1)    #[(bsz*src_len), emb_dim]
        # FWD
        src_remap = self.linear(src_reshape)                    #[(bsz*src_len), hdim]
        # correct shape for StackedCNN
        src_remap = src_remap.view(src.size(0), src.size(1), -1) #[bsz, src_len, hdim]
        src = shape_transform(src_remap)                         #[bsz,hdim,src_len,1]
         
        # StackedCNN
        cnn_out = self.cnn(src)      #[bsz,input_hsz,src_len,1]
        # reshape for trf layers:
        cnn_out = cnn_out.squeeze(3).transpose(1,2).contiguous() # [bsz, src_len, emb_dim]
        # TRF                                                          
        for layer in self.transformer:
            out = layer(cnn_out, mask=None)
        out = self.layer_norm(out)
        # var to init decoder state
        state = out.new_full(out.shape, 0) # THIS IS A DUMMY - TRF DECODERS DON'T NEED INITIALIZATION
        # return enc_final, memory_bank, lengths
        return state, out.transpose(0, 1).contiguous(), orig_lengths.new_tensor(lengths)

        

    # CNN
    def cnnforward(self, input, lengths=None, hidden=None):
        """See :class:`onmt.modules.EncoderBase.forward()`"""
        import ipdb; ipdb.set_trace()
        self._check_args(input, lengths, hidden)

        emb = self.embeddings(input)                             #[src_len, bsz, emb_dim]
        emb = emb.transpose(0, 1).contiguous()                   #[bsz, src_len, emb_dim]
        emb_reshape = emb.view(emb.size(0) * emb.size(1), -1)    #[(bsz*src_len), emb_dim]
        emb_remap = self.linear(emb_reshape)                     #[(bsz*src_len), emb_dim]
        emb_remap = emb_remap.view(emb.size(0), emb.size(1), -1) #[bsz, src_len, emb_dim]
        emb_remap = shape_transform(emb_remap)                   #[bsz, emb_dim, src_len, 1]
        out = self.cnn(emb_remap)                                #[bsz, emb_dim, src_len, 1]

        return emb_remap.squeeze(3).transpose(0, 1).contiguous(), \
            out.squeeze(3).transpose(0, 1).contiguous(), lengths    #[emb_dim,bsz,src_len],[emb_dim,bsz,src_len], [bsz]

    # TRF:
    def trfforward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        import ipdb; ipdb.set_trace()
        self._check_args(src, lengths)

        emb = self.embeddings(src) # embeddings_layer: [vocabsz] -> [rnn_size]
                                   # dim(emb) = [src_len, bsz, emb_dim]

        out = emb.transpose(0, 1).contiguous() # [bsz, src_len, emb_dim]
        words = src[:, :, 0].transpose(0, 1)
        w_batch, w_len = words.size()
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1)  # [B, 1, T]
        # Run the forward pass of every layer of the tranformer.
        for layer in self.transformer:
            out = layer(out, mask)
        out = self.layer_norm(out)

        return emb, out.transpose(0, 1).contiguous(), lengths



    def update_dropout(self, dropout):
        self.dropout.p = dropout
        for i in range(self.enc_layers - 1):
            getattr(self, 'rnn_%d' % i).dropout = dropout