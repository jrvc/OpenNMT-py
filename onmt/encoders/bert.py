"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn

from onmt.encoders.encoder import EncoderBase
from onmt.modules import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.modules.position_ffn import ActivationFunction
from onmt.utils.misc import sequence_mask
from onmt.utils.logging import logger

from transformers import (
    BertConfig, 
    BertForMaskedLM, 
    BertModel, 
    BertTokenizer,
    )

class BertEncoder(EncoderBase):
    """BERT encoder from huggingface Transformers
    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings
        pos_ffn_activation_fn (ActivationFunction):
            activation function choice for PositionwiseFeedForward layer

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """
    def __init__(
        self, 
        vocab, 
        bert_type='bert-base-uncased', 
        outdim=512, 
        output_hidden_states=False,
        return_dict=True,
        freeze_bert=False,
    ):
        super(BertEncoder, self).__init__()

        # BERT: 
        config = BertConfig.from_pretrained(bert_type, output_hidden_states=output_hidden_states)
        self.bert = BertModel.from_pretrained(bert_type, config=config)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_type)
        
        self.output_hidden_states = output_hidden_states
        self.return_dict = return_dict
        # LINEAR
        self.bert_encdim = self.bert.pooler.dense.in_features
        self.linear = nn.Linear(self.bert_encdim, outdim)
        self.vocab = vocab

        if freeze_bert:
            logger.info(f'ATTENTION: BERT parameters are not to be optimized during training.')
            for param in self.bert.parameters():
                param.requires_grad = False

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            vocab=embeddings,
            bert_type='bert-base-uncased', 
            outdim=opt.rnn_size,  
            output_hidden_states=True,
            return_dict=True,
            freeze_bert=opt.freeze_bert,
        )

    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""

    
        #self, input_ids, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=False, **kwargs
    
        '''
        runs bert and a linear projection
        '''            
        self._check_args(src, lengths)
        
        src_texts = self.recover_sents(src, lengths)

        encoded_sentences = self.bert_tokenizer(src_texts, padding=True, return_tensors='pt').to(src.device)

        new_lengths = encoded_sentences['attention_mask'].sum(axis=1)
        new_src = encoded_sentences['input_ids'].transpose(0,1).unsqueeze(dim=2)
        
        encoder_outputs = self.bert(encoded_sentences['input_ids'], 
                                attention_mask=encoded_sentences['attention_mask'],
                                return_dict=self.return_dict, 
                                output_hidden_states=self.output_hidden_states , 
                                )#**kwargs)
        
        #linear projection
        if self.return_dict:
            hstates = self.linear(encoder_outputs['last_hidden_state']) # [bsz, sentlen, mt_hdim]
            encoder_outputs['last_hidden_state'] = hstates
        else:
            hstates = self.linear(encoder_outputs[0]) # [bsz, sentlen, mt_hdim]
            encoder_outputs = (hstates,) + encoder_outputs[1:]
        


        emb = encoder_outputs['hidden_states'][0].transpose(0,1).contiguous()
        out = encoder_outputs['last_hidden_state'].transpose(0,1).contiguous()

        #return enc_state , memory_bank, lengths 
        return emb, out, (new_lengths,new_src) # [sentlen, bsz, rnn_size], [sentlen, bsz, rnn_size], [bsz]


    def recover_sents(self, src, lengths):
        # TODO: decode in a better way than just ' '.join(list(tokens))
        #return [ ' '.join([self.vocab.itos[idx] for i,idx in enumerate(sent) if i < sl ]) for sent,sl in zip(src.squeeze().T,lengths )]
        src_texts=[]
        for sent, sl in zip(src.squeeze(-1).T, lengths):    
            text = ' '.join([self.vocab.itos[idx] for i,idx in enumerate(sent) if i < sl ])
            src_texts.append(text)
        return src_texts
        

