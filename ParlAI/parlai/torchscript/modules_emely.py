#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import List, Dict, Optional, Tuple

import torch.jit
from torch import nn as nn

from parlai.core.dict import DictionaryAgent
from parlai.core.torch_agent import TorchAgent
from parlai.utils.bpe import SubwordBPEHelper


class TorchScriptGreedySearch(nn.Module):
    """
    A helper class for exporting simple greedy-search models via TorchScript.

    Models with extra inputs will need to override to include more variables.
    """

    # We currently only support these specific dictionary settings
    CAIRAOKE_DICT_PARAMS = {
        "dict_class": "parlai.core.dict:DictionaryAgent",
        "dict_initpath": None,
        "dict_language": "english",
        "dict_max_ngram_size": -1,
        "dict_minfreq": 0,
        "dict_maxtokens": -1,
        "dict_tokenizer": "bpe",
        "dict_lower": True,
        "dict_textfields": "text,labels",
        "dict_loaded": True,
        'bpe_debug': False,
    }

    def __init__(self, agent: TorchAgent):
        super().__init__()

        # Dictionary/tokenization setup
        for key, val in self.CAIRAOKE_DICT_PARAMS.items():
            assert (
                agent.opt.get(key, val) == val
            ), f'The only currently supported value of "{key}" is {val}!'
        orig_dict: DictionaryAgent = agent.dict
        orig_bpe: SubwordBPEHelper = orig_dict.bpe

        # Cast the values as floats to be able to compare to float('inf') when doing BPE
        # splitting
        self.dict = ScriptableDictionaryAgent(
            null_token=orig_dict.null_token,
            end_token=orig_dict.end_token,
            unk_token=orig_dict.unk_token,
            start_token=orig_dict.start_token,
            freq=orig_dict.freq,
            tok2ind=orig_dict.tok2ind,
            ind2tok=orig_dict.ind2tok,
            bpe_add_prefix_space=agent.opt['bpe_add_prefix_space'],
            bpe_codes=orig_bpe.bpe_codes,
            separator=orig_bpe.separator,
            temp_separator=agent.opt["temp_separator"],
        )

        # History tracking and start/end tokens
        self.delimiter_tok = agent.history.delimiter_tok
        self.history_size = agent.opt['history_size']
        if agent.opt.get('history_add_global_end_token', None) is not None:
            self.global_end_token = agent.dict[agent.dict.end_token]
        else:
            self.global_end_token = None
        self.text_truncate = agent.opt.get('text_truncate') or agent.opt['truncate']
        self.text_truncate = self.text_truncate if self.text_truncate >= 0 else None

        self.start_idx = agent.model.START_IDX
        self.end_idx = agent.model.END_IDX
        self.null_idx = agent.model.NULL_IDX
        self.initial_decoder_input = [self.end_idx, self.start_idx]

        agent.model.eval()

        # Create versions of the model and decoder that will flatten the incremental
        # state dict, as required by TorchScript
        wrapped_decoder = DecoderIncrStateFlattener(agent.model.decoder)
        wrapped_model = ModelIncrStateFlattener(agent.model)

        # Create sample inputs for tracing
        sample_tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        encoder_states = agent.model.encoder(sample_tokens)
        initial_generations = self._get_initial_decoder_input(sample_tokens)
        latent, initial_incr_state = wrapped_decoder(
            initial_generations, encoder_states
        )
        logits = agent.model.output(latent[:, -1:, :])
        _, preds = logits.max(dim=2)
        incr_state = {k: torch.clone(v) for k, v in initial_incr_state.items()}
        # Copy the initial incremental state, used when tracing the
        # .reorder_decoder_incremental_state() method below, to avoid having it be
        # mutated by the following line
        incr_state = wrapped_model.reorder_decoder_incremental_state(
            incr_state, torch.tensor([0], dtype=torch.long, device=sample_tokens.device)
        )
        generations = torch.cat([initial_generations, preds], dim=1)

        # Do tracing
        self.encoder = torch.jit.trace(agent.model.encoder, sample_tokens)
        self.decoder_first_pass = torch.jit.trace(
            wrapped_decoder, (initial_generations, encoder_states), strict=False
        )
        # We do strict=False to avoid an error when passing a Dict out of
        # decoder.forward()
        self.partially_traced_model = torch.jit.trace_module(
            wrapped_model,
            {
                'output': (latent[:, -1:, :]),
                'reorder_decoder_incremental_state': (
                    initial_incr_state,
                    torch.tensor([0], dtype=torch.long, device=sample_tokens.device),
                ),
            },
            strict=False,
        )
        print(generations.size())
        print(len(encoder_states))
        print(incr_state)
        self.decoder_later_pass = torch.jit.trace(
            wrapped_decoder, (generations, encoder_states, incr_state), strict=False
        )

    def _get_initial_decoder_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Workaround because we can't use TGM._get_initial_decoder_input() directly.

        When we try to call that function, we get a "RuntimeError: Type 'Tuple[int,
        int]' cannot be traced. Only Tensors and (possibly nested) Lists, Dicts, and
        Tuples of Tensors can be traced" error.
        """
        bsz = x.size(0)
        return (
            torch.tensor(self.initial_decoder_input, dtype=torch.long)
            .expand(bsz, len(self.initial_decoder_input))
            .to(x.device)
        )

    def parse(self, text: str) -> List[int]:
        return self.dict.txt2vec(text)

    def _v2t(self, vec: List[int]) -> str:
        """
        Convert token indices to string of tokens.
        """
        new_vec: List[int] = []
        for i in vec:
            if i == self.end_idx:
                break
            elif i != self.start_idx:
                new_vec.append(i)
        return self.dict.vec2txt(new_vec)

    def forward(self, context: str, max_len: int = 128) -> str:

        # Vectorize all lines of context
        history_vecs: List[List[int]] = []
        context_lines = context.split('\n')
        if self.history_size > 0:
            context_lines = context_lines[-self.history_size :]
        for line in context_lines:
            history_vecs.append(self.parse(line))

        # Get full history vec
        text_vecs: List[List[int]] = []
        for vec in history_vecs[:-1]:
            text_vecs += [vec]
            text_vecs += [self.delimiter_tok]
        text_vecs += [history_vecs[-1]]
        if self.global_end_token is not None:
            text_vecs += [[self.global_end_token]]

        # Flatten text_vecs
        flattened_text_vec: List[int] = []
        for vec in text_vecs:
            for token in vec:
                flattened_text_vec.append(token)

        # Format history vec given various logic
        if self.text_truncate is not None:
            truncate_length = self.text_truncate - 2
            if len(flattened_text_vec) > truncate_length:
                flattened_text_vec = flattened_text_vec[-truncate_length:]
        flattened_text_vec = torch.tensor(flattened_text_vec, dtype=torch.long)
        # originally "if is_bart:"
        flattened_text_vec = torch.cat(
            [
                torch.tensor([self.start_idx], dtype=torch.long),
                flattened_text_vec,
                torch.tensor([self.end_idx], dtype=torch.long),
            ],
            dim=0,
        )

        # Pass through the encoder and decoder to generate tokens
        batch_text_vec = torch.unsqueeze(flattened_text_vec, dim=0)  # Add batch dim
        encoder_states = self.encoder(batch_text_vec)
        generations = self._get_initial_decoder_input(batch_text_vec)
        # keep track of early stopping if all generations finish
        seen_end = torch.zeros(
            batch_text_vec.size(0), device=batch_text_vec.device, dtype=torch.bool
        )
        incr_state: Dict[str, torch.Tensor] = {}
        for token_idx in range(max_len):
            if token_idx == 0:
                latent, incr_state = self.decoder_first_pass(
                    generations, encoder_states
                )
            else:
                latent, incr_state = self.decoder_later_pass(
                    generations, encoder_states, incr_state
                )
            logits = self.partially_traced_model.output(latent[:, -1:, :])
            _, preds = logits.max(dim=2)
            incr_state = self.partially_traced_model.reorder_decoder_incremental_state(
                incr_state,
                torch.tensor([0], dtype=torch.long, device=batch_text_vec.device),
            )
            seen_end = seen_end + (preds == self.end_idx).squeeze(1)
            generations = torch.cat([generations, preds], dim=1)
            if torch.all(seen_end):
                break

        # Get the label from the generated tokens and update the history
        generation_tokens: List[int] = generations[0].tolist()
        label = self._v2t(generation_tokens)

        return label


# class TorchScriptBeamSearch(nn.Module):
#     """
#     A helper class for exporting simple beam-search models via TorchScript.

#     Models with extra inputs will need to override to include more variables.
#     """


class BaseIncrStateFlattener(nn.Module):
    """
    Flatten/unflatten the incremental state for use with TorchScripting.

    Typically, the incremental state will be stored as a Dict[int, Dict[str, Dict[str,
    torch.Tensor]]], where the 3 dictionary levels map decoder layer, attention type,
    and previous key/value/mask, respectively. However, TorchScript expects dicts to be
    of type Dict[str, torch.Tensor], and thus all input incremental states when
    TorchScripting will have to be of that type. We thus unflatten the input incremental
    state, already of type Dict[str, torch.Tensor], to pass it into whatever method
    needs it, and we flatten it again after the updated incremental state is passed back
    out.

    This is a base class that provides methods for flattening/unflattening: subclasses
    will call these methods as the incremental state is passed into and out of their own
    methods.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def _unflatten_incr_state(
        self, flat_incr_state: Dict[str, torch.Tensor]
    ) -> Dict[int, Dict[str, Dict[str, torch.Tensor]]]:
        """
        Unflatten the input incremental state.

        For instance, flat_incr_state['layer_0__self_attn__prev_key'] will be stored in
        structured_incr_state[0]['self_attn']['prev_key'].
        """
        structured_incr_state = defaultdict(lambda: defaultdict(dict))
        for key, state in flat_incr_state.items():
            layer_idx_str, attn_type, state_type = key.split('__')
            structured_incr_state[int(layer_idx_str)][attn_type][state_type] = state
        return dict({k: dict(v) for k, v in structured_incr_state.items()})
        # Turn the nested defaultdicts back into regular dicts

    def _flatten_incr_state(
        self, structured_incr_state: Dict[int, Dict[str, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Flatten the input incremental state.

        For instance, structured_incr_state[0]['self_attn']['prev_key'] will be stored
        in flat_incr_state['layer_0__self_attn__prev_key'].
        """
        flat_incr_state = {}
        for layer_idx, dict1 in structured_incr_state.items():
            for attn_type, dict2 in dict1.items():
                for state_type, state in dict2.items():
                    key = f'{layer_idx:d}__{attn_type}__{state_type}'
                    flat_incr_state[key] = state
        return flat_incr_state


class DecoderIncrStateFlattener(BaseIncrStateFlattener):
    """
    Wrapper for a TransformerDecoder that will unflatten/flatten the incremental state.

    Unflattening/flattening will occur before passing the incremental state into and out
    of .forward().
    """

    def forward(
        self,
        input_: torch.LongTensor,
        encoder_state: Tuple[torch.Tensor, torch.Tensor],
        flat_incr_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if flat_incr_state is not None:
            structured_incr_state = self._unflatten_incr_state(flat_incr_state)
        else:
            structured_incr_state = None
        tensor, new_structured_incr_state = self.module.forward(
            input=input_, encoder_state=encoder_state, incr_state=structured_incr_state
        )
        new_flat_incr_state = self._flatten_incr_state(new_structured_incr_state)
        return tensor, new_flat_incr_state


class ModelIncrStateFlattener(BaseIncrStateFlattener):
    """
    Wrapper for a TransformerGeneratorModel to unflatten/flatten the incremental state.

    Unflattening/flattening will occur before passing the incremental state into and out
    of .reorder_decoder_incremental_state(). We also support .output(), which is also
    traced.
    """

    def reorder_decoder_incremental_state(
        self, flat_incr_state: Dict[str, torch.Tensor], inds: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        structured_incr_state = self._unflatten_incr_state(flat_incr_state)
        new_structured_incr_state = self.module.reorder_decoder_incremental_state(
            incremental_state=structured_incr_state, inds=inds
        )
        return self._flatten_incr_state(new_structured_incr_state)

    def output(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.module.output(tensor)

@torch.jit.script
class ScriptableSubwordBpeHelper(object):
    """
    Version of parlai.utils.bpe.SubwordBpeHelper that can be TorchScripted.
    """

    @classmethod
    def findall(cls, text: str) -> List[str]:
        """
        Split tokens in a manner that replicates parlai.utils.bpe.SubwordBpeHelper.
        """

        tokens: List[str] = []
        idx = 0
        num_passes = 0
        while idx < len(text):
            num_passes += 1
            if not text[idx].isspace():
                last_matching_idx = idx
                if text[idx].isalpha() or text[idx]=='_' or text[idx].isnumeric():
                    while (
                        last_matching_idx + 1 < len(text)
                        and (text[last_matching_idx + 1].isalpha()
                        or text[last_matching_idx + 1]=='_' 
                        or text[last_matching_idx + 1].isnumeric())
                    ):
                        last_matching_idx += 1
                else:
                    while (
                        last_matching_idx + 1 < len(text)
                        and not text[last_matching_idx + 1].isspace()
                        and not text[last_matching_idx + 1].isalpha()
                        and not text[last_matching_idx + 1].isnumeric()
                    ):
                        last_matching_idx += 1
                tokens.append(text[idx:last_matching_idx + 1])
                idx = last_matching_idx + 1
            else:
                idx = idx + 1
        return tokens

    def __init__(
        self,
        add_prefix_space: bool,
        bpe_codes: Dict[str, int],
        separator: str,
        temp_separator: str,
    ):

        self.add_prefix_space: bool = add_prefix_space
        self.bpe_codes: Dict[str, int] = bpe_codes
        self.separator: str = separator
        self.temp_separator: str = temp_separator

    def encode(self, text: str) -> List[str]:
        """
        Tokenize text.

        Checks for add_prefix_space; handles accordingly.

        :param text:
            text to tokenize

        :return tokens:
            A list of tokens
        """
        if self.add_prefix_space:
            text = f' {text}'
        return self.helper_encode(text)


    def helper_encode(self, text: str) -> List[str]:
        """
        Tokenize text.

        :param text:
            text to tokenize

        :return tokens:
            A list of tokens
        """
        text = text.replace('\n', ' __newln__ ')
        #return self.findall(text)
        return self.segment_tokens(self.findall(text))
    
    def segment_tokens(self, tokens: List[str]) -> List[str]:
        """segment a sequence of tokens with BPE encoding"""
        output: List[str] = []
        for word in tokens:
            # eliminate double spaces
            if not word:
                continue
            new_word: List[str] = self.encode_token(word)

            for item in new_word[:-1]:
                output.append(item + self.separator)
            output.append(new_word[-1])

        return output
    
    def encode_token(self, orig: str) -> List[str]:
        """Encode word based on list of BPE merge operations, which are applied consecutively"""
        """Only implemented for version (0,2) in codec file"""

        if len(orig) == 1:
            return [orig]

        word: List[str] = list(orig[:-1]) + [orig[-1] + '</w>']
        q = 1
        while len(word) > 1:

            # get list of symbol pairs; optionally apply dropout
            ranks: List[int] = []
            codes: List[str] = []
            idxs: List[int] = []
            
            for (i,pair) in enumerate(zip(word, word[1:])):
                if self.temp_separator.join(pair) in self.bpe_codes:
                    ranks.append(self.bpe_codes[self.temp_separator.join(pair)])
                    codes.append(self.temp_separator.join(pair))
                    idxs.append(i)
            if len(ranks)==0:
                break

            #get first merge operation in list of BPE codes
            min_rank_idx: int = ranks.index(min(ranks))
            bigram: str = codes[min_rank_idx]
            
            # find start position of all pairs that we want to merge
            positions: List[int] = []
            for i in range(0,len(ranks)):
                if codes[i]==bigram:
                    positions.append(idxs[i])

            i = 0
            new_word: List[str] = []
            bigram = bigram.replace('__space__','')
            for j in positions:
                # merges are invalid if they start before current position. This can happen if there are overlapping pairs: (x x x -> xx x)
                if j < i:
                    continue
                new_word.extend(word[i:j]) # all symbols before merged pair
                new_word.append(bigram) # merged pair
                i = j+2 # continue after merged pair
            new_word.extend(word[i:]) # add all symbols until end of word
            word = new_word

        # don't print end-of-word symbols
        if word[-1] == '</w>':
            word = word[:-1]
        elif word[-1].endswith('</w>'):
            word[-1] = word[-1][:-4]

        return word

    def decode(self, tokens: List[str]) -> str:
        """
        Decode list of tokens into a text string.

        :param tokens:
            list of tokens

        :return text:
            decoded text
        """
        text = self.helper_decode(tokens)
        if self.add_prefix_space:
            assert text.startswith(' ')
            text = text.lstrip(' ')
        return text

    def helper_decode(self, tokens: List[str]) -> str:
        """
        Decode list of tokens into text string.

        :param tokens:
            list of tokens

        :return:
            decoded text
        """
        text = ' '.join(tokens)
        text = text.replace('@@ ', '')
        # It's also possible that we get a BPE encoding on the end of the word
        if text.endswith('@@'):
            text = text[:-2]
        return text


@torch.jit.script
class ScriptableDictionaryAgent:
    """
    Builds and/or loads a dictionary.

    All code is TorchScriptable.
    """

    def __init__(
        self,
        null_token: str,
        end_token: str,
        unk_token: str,
        start_token: str,
        freq: Dict[str, int],
        tok2ind: Dict[str, int],
        ind2tok: Dict[int, str],
        bpe_add_prefix_space: bool,
        bpe_codes: Dict[str, int],
        separator: str,
        temp_separator: str,
    ):

        self.null_token = null_token
        self.end_token = end_token
        self.unk_token = unk_token
        self.start_token = start_token

        self.freq = freq
        self.tok2ind = tok2ind
        self.ind2tok = ind2tok

        # cache unk token for later
        self._unk_token_idx = self.tok2ind[self.unk_token]

        # Initialize tokenizer
        self.bpe = ScriptableSubwordBpeHelper(
            add_prefix_space=bpe_add_prefix_space,
            bpe_codes=bpe_codes,
            separator=separator,
            temp_separator=temp_separator
        )

    def _word_lookup(self, key: str) -> int:
        """
        Return index from token, or unk_token's index, or None.
        """
        if key in self.tok2ind:
            return self.tok2ind[key]
        else:
            return self._unk_token_idx

    def _index_lookup(self, key: int) -> str:
        """
        Return token from index, or unk_token.
        """
        if key in self.ind2tok:
            return self.ind2tok[key]
        else:
            return self.unk_token

    def tokenize(self, text: str) -> List[str]:
        """
        Return a sequence of tokens from the iterable.

        Also handles special tokens for some tokenizers
        """
        
        # calls the selected tokenizer function e.g. 're' => re_tokenize(text)
        word_tokens = self.bpe_tokenize(text)

        return word_tokens

    def bpe_tokenize(self, text: str) -> List[str]:
        """
        Return a sequence of BPE-tokens from the text.
        """
        return self.bpe.encode(text)

    def txt2vec(self, text: str) -> List[int]:
        """
        Convert a string to a vector (list of ints).

        First runs a sentence tokenizer, then a word tokenizer.
        """
        itr: List[int] = []
        for token in self.tokenize(str(text)):
            itr.append(self._word_lookup(token))
        return itr

    def vec2txt(self, vector: List[int]) -> str:
        """
        Convert a vector of IDs to a string.

        Converts a vector (iterable of ints) into a string, with each token separated by
        the delimiter (default ``' '``).
        """
        tokens = [self._index_lookup(idx) for idx in vector]
        text = self.bpe.decode(tokens)
        return text
