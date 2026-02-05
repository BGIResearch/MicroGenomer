from typing import Sequence, Tuple, List, Union
from abc import ABC
from abc import abstractmethod
import logging
import itertools
from transformers import PreTrainedTokenizer
import torch
import numpy as np

logger = logging.getLogger(__name__)

class BaseLevelTokenizer(object):
    """
    Tokenizer for DNA Base Level Tokenization.
    """
    def __init__(self, **kwargs):
        super(BaseLevelTokenizer, self).__init__()
        
        ### Set normal tokens
        self.all_toks = ['[pad]', 'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-']
        
        ### Set special tokens
        _special_tokens = ['tMASK', 'gMASK', 'sMASK', 'eod', 'sop', 'eop', '</s>', 'HC','LC', 'HUMAN']
        self.special_tokens = { tok: len(self.all_toks)+i for i,tok in enumerate(_special_tokens) }
        self.special_tokens_decoder = { v:k for k, v in self.special_tokens.items() }
        self.special_tokens['eos'] = self.special_tokens['</s>']
        self.all_toks.extend(_special_tokens)
        
        self._DNA_tokens = ['1', '2', '3', '4', '5']
        self.all_toks.extend(self._DNA_tokens) 

        self.vocab = {tok:idx for idx,tok in enumerate(self.all_toks)}
        self.command_token = {'[MASK]':'MASK', '[gMASK]': 'gMASK', '[sMASK]':'sMASK'}
        
        self.gMASK_token_id = self.convert_token_to_id('gMASK')
        self.sop_token_id   = self.convert_token_to_id('sop')
        self.eos_token_id   = self.convert_token_to_id('</s>')
        self.pad_token_id   = self.convert_token_to_id('[pad]')
    
    def __len__(self):
        return len(self.vocab)
    
    def get_special_token(self, token):
        return self.special_tokens[token]
    
    def get_vocab(self):
        return self.vocab
    
    def convert_id_to_token(self, idx):
        idx = int(idx)
        if idx == 0:
            return '[pad]'
        elif idx in self.special_tokens_decoder:
            return f"[{self.special_tokens_decoder[idx]}]"
        else:
            return self.all_toks[idx]
    
    def convert_token_to_id(self, token):
        if token == '[pad]':
            return 0
        elif token in self.special_tokens:
            return self.special_tokens[token]
        else:
            return self.vocab[token]
    
    def encode(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        return ids
    
    def decode(self, Ids):
        return ''.join([self.convert_id_to_token(tok) for tok in Ids])
    
    def tokenize(self, text):
        tokens = []
        i = 0
        special_bracket = { '[': ']', '<': '>' }
        while i<len(text):
            if text[i] == ' ':
                pass
            elif text[i] in special_bracket:
                exp_close_bracket = special_bracket[ text[i] ]
                special_token = ''
                i += 1
                while i<len(text) and text[i]!=exp_close_bracket:
                    special_token += text[i]
                    i += 1
                if special_token == 'pad':
                    tokens.append( self.vocab['[pad]'] )
                else:
                    assert special_token in self.special_tokens, f"Unexpected special token: {special_token}"
                    tokens.append( self.vocab[ special_token ] )
                special_token = None
            else:
                tokens.append( self.vocab[ text[i] ] )
            i += 1
        return tokens



class DNATokenizer(PreTrainedTokenizer):
    """
    DNA Tokenizer based on base level tokenizer
    """

    def __init__(
        self,
        vocab_file='xxx',
        padding_side="right",
        clean_up_tokenization_spaces=False,
        encode_special_tokens=True,
        **kwargs
    ):
        self.name = "DNATokenizer"
        self.vocab_file = vocab_file
        self.tokenizer = BaseLevelTokenizer()
        self.special_tokens = self.tokenizer.special_tokens
        self.encode_special_tokens = encode_special_tokens
        
        super().__init__(
            padding_side=padding_side,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs
        )
    
    def get_command(self, token):
        if token in self.special_tokens:
            return self.special_tokens[token]
        assert token in self.tokenizer.special_tokens, f"{token} is not a special token for {self.name}"
        return self.tokenizer.special_tokens[token]
    
    @property
    def unk_token(self) -> str:
        return '[pad]'
    
    @property
    def pad_token(self) -> str:
        return '[pad]'

    @property
    def eos_token(self) -> str:
        return '</s>'

    @property
    def unk_token_id(self) -> int:
        return '[pad]'

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def gMASK_token_id(self):
        return self.tokenizer.gMASK_token_id

    @property
    def sop_token_id(self):
        return self.tokenizer.sop_token_id

    @property
    def id_token_id(self):
        return self.tokenizer.id_token_id

    def IdToToken(self, id_):
        return self.tokenizer.convert_id_to_token(id_)

    def TokenToId(self, token):
        return self.tokenizer.convert_token_to_id(token)

    @unk_token.setter
    def unk_token(self, value):
        logger.warning("Setting unk_token is not supported, use the default one.")

    @pad_token.setter
    def pad_token(self, value):
        logger.warning("Setting pad_token is not supported, use the default one.")

    @eos_token.setter
    def eos_token(self, value):
        logger.warning("Setting eos_token is not supported, use the default one.")

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)
    
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.tokenizer.convert_id_to_token(index)

    def get_vocab(self):
        """ Returns vocab as a dict """
        vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}
        return vocab
    
    @property
    def eod(self):
        return self.tokenizer.get_special_token('eos')

    def detokenize(self, Ids, type_token=False):
        new_tokens = self.tokenizer.decode(Ids)
        return new_tokens

    def tokenize(self, text):
        ids = self.tokenizer.tokenize(text)
        return ids
    
    def build_chat_input(self, query):
        input_ids  = [ self.tokenizer.convert_token_to_id('gMASK'), self.tokenizer.convert_token_to_id('sop') ]
        input_ids += [ self.tokenizer.convert_token_to_id(tok) for tok in query ]
        input_ids += [ self.tokenizer.convert_token_to_id('ID') ]
        # return self.batch_encode_plus([input_ids], return_tensors="pt", is_split_into_words=True)
        
        position_ids = torch.stack([torch.zeros(len(input_ids)), torch.arange(len(input_ids))], axis=0).unsqueeze(0).long()
        return {
            'input_ids': torch.from_numpy(np.array([ input_ids ])).long(),
            'attention_mask': None,
            'position_ids': position_ids
        }

