#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import List, Dict, Optional, Tuple

import torch.jit
from torch import nn as nn
import torch.nn.functional as F

from parlai.core.dict import DictionaryAgent
from parlai.core.torch_agent import TorchAgent
from parlai.utils.bpe import SubwordBPEHelper


class TorchScriptedEmelyAgent(nn.Module):
    """
    A helper class for exporting Emely via TorchScript.
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
            dict_lower=orig_dict.lower,
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
        self.initial_decoder_input = [self.start_idx]#[self.end_idx, self.start_idx]

        self.inference = agent.opt.get("inference","greedy")
        self.temperature: float = agent.temperature
        self.beam_size: int = agent.opt.get("beam_size",1)
        self.block_ngram = agent.beam_block_ngram
        self.context_block_ngram = agent.beam_context_block_ngram
        self.padding_token = agent.NULL_IDX
        self.bos_token = agent.START_IDX
        self.eos_token = agent.END_IDX
        if self.inference=="greedy":
            self.min_length = 0
        else:
            self.min_length = agent.beam_min_length
        self.length_penalty = agent.opt.get('beam_length_penalty', 0.65)

        agent.model.eval()

        # Create versions of the model and decoder that will flatten the incremental
        # state dict, as required by TorchScript
        wrapped_decoder = DecoderIncrStateFlattener(agent.model.decoder)
        wrapped_model = ModelIncrStateFlattener(agent.model)

        # Create sample inputs for tracing
        sample_tokens = torch.tensor([[1, 2, 3, 4, 5] for _ in range(self.beam_size)], dtype=torch.long)
        encoder_states = agent.model.encoder(sample_tokens)
        initial_decoder_input = self._get_initial_decoder_input(self.beam_size)
        score, initial_incr_state = wrapped_decoder(
            initial_decoder_input, encoder_states
        )
        preds = agent.model.output(score[:, -1:, :])
        _, preds = preds.max(dim=2)
        incr_state = {k: torch.clone(v) for k, v in initial_incr_state.items()}
        # Copy the initial incremental state, used when tracing the
        # .reorder_decoder_incremental_state() method below, to avoid having it be
        # mutated by the following line
        incr_state = wrapped_model.reorder_decoder_incremental_state(
            incr_state, torch.tensor([0], dtype=torch.long).expand(self.beam_size)
        )
        decoder_input = torch.cat([initial_decoder_input, preds], dim=1)

        # Do tracing
        self.encoder = torch.jit.trace(agent.model.encoder, sample_tokens)
        self.decoder_first_pass = torch.jit.trace(
            wrapped_decoder, (initial_decoder_input, encoder_states), strict=False
        )
        # We do strict=False to avoid an error when passing a Dict out of
        # decoder.forward()
        self.partially_traced_model = torch.jit.trace_module(
            wrapped_model,
            {
                'output': (score[:, -1:, :]),
                'reorder_decoder_incremental_state': (
                    initial_incr_state,
                    torch.tensor([0], dtype=torch.long).expand(self.beam_size),
                ),
            },
            strict=False,
        )
        self.decoder_later_pass = torch.jit.trace(
            wrapped_decoder, (decoder_input, encoder_states, incr_state), strict=False
        )

    def _get_initial_decoder_input(self, beam_size: int) -> torch.Tensor:
        """
        Workaround because we can't use TGM._get_initial_decoder_input() directly.

        When we try to call that function, we get a "RuntimeError: Type 'Tuple[int,
        int]' cannot be traced. Only Tensors and (possibly nested) Lists, Dicts, and
        Tuples of Tensors can be traced" error.
        """
        return (
            torch.tensor(self.initial_decoder_input, dtype=torch.long)
            .expand(beam_size,1)
            # torch.tensor(self.initial_decoder_input, dtype=torch.long)
            # .expand(1, len(self.initial_decoder_input))
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
        # originally "if is_bart: Seems to be excluded in Emely"
        # flattened_text_vec = torch.cat(
        #     [
        #         torch.tensor([self.start_idx], dtype=torch.long),
        #         flattened_text_vec,
        #         torch.tensor([self.end_idx], dtype=torch.long),
        #     ],
        #     dim=0,
        # )

        # Pass through the encoder and decoder to generate tokens
        batch_text_vec = torch.unsqueeze(flattened_text_vec, dim=0).repeat(self.beam_size,1)  # Add batch dim
        encoder_states = self.encoder(batch_text_vec)
        decoder_input = self._get_initial_decoder_input(self.beam_size)
        # keep track of early stopping if all generations finish

        beam = ScriptableTreeSearch(self.beam_size,
                                    self.inference,
                                    block_ngram = self.block_ngram,
                                    context_block_ngram = self.context_block_ngram,
                                    padding_token = self.padding_token,
                                    bos_token = self.bos_token,
                                    eos_token = self.eos_token,
                                    min_length = self.min_length,
                                    length_penalty = self.length_penalty,
                                    )
        beam.set_context(flattened_text_vec)
        
        incr_state: Dict[str, torch.Tensor] = {}
        for token_idx in range(max_len):
            if beam.is_done():
                break

            if token_idx == 0:
                score, incr_state = self.decoder_first_pass(
                    decoder_input, encoder_states
                )
            else:
                score, incr_state = self.decoder_later_pass(
                    decoder_input, encoder_states, incr_state
                )
            score = score[:, -1:, :]
            score = self.partially_traced_model.output(score)
            score = score.view(1, self.beam_size, -1)
            if self.temperature != 1.0:
                score.div_(self.temperature)
            #_, preds = score.max(dim=2)    # Previous
            score = F.log_softmax(score, dim=-1, dtype=torch.float32)
            if not beam.is_done():
                beam.advance(score[0])
            incr_state_inds = beam.get_backtrack_from_current_step()
            incr_state = self.partially_traced_model.reorder_decoder_incremental_state(
                incr_state,incr_state_inds
            )
            selection = beam.get_output_from_current_step().unsqueeze(-1)
            decoder_input = self._get_next_decoder_input(
                decoder_input, selection, incr_state_inds
            )

        best = beam.get_rescored_finished()
        bestlist: List[int] = [int(best[i].item()) for i in range(best.size()[0])]
        label = self._v2t(bestlist)

        return label

    def _get_next_decoder_input(
        self,
        prev_input: torch.LongTensor,
        selection: torch.LongTensor,
        incr_state_inds: torch.LongTensor,
    ) -> torch.LongTensor:
        """
        Return next decoder input.

        :param prev_input:
            previous input to decoder
        :param selection:
            token selections for current timestep
        :param inds:
            incremental state indices

        :return decoder input:
            return decoder input for next timestep
        """
        prev_input = torch.index_select(prev_input, 0, incr_state_inds)
        decoder_input = torch.cat([prev_input, selection], dim=-1)
        return decoder_input

    # """Original forward function"""
    # def forward(self, context: str, max_len: int = 128) -> str:

    #     # Vectorize all lines of context
    #     history_vecs: List[List[int]] = []
    #     context_lines = context.split('\n')
    #     if self.history_size > 0:
    #         context_lines = context_lines[-self.history_size :]
    #     for line in context_lines:
    #         history_vecs.append(self.parse(line))

    #     # Get full history vec
    #     text_vecs: List[List[int]] = []
    #     for vec in history_vecs[:-1]:
    #         text_vecs += [vec]
    #         text_vecs += [self.delimiter_tok]
    #     text_vecs += [history_vecs[-1]]
    #     if self.global_end_token is not None:
    #         text_vecs += [[self.global_end_token]]

    #     # Flatten text_vecs
    #     flattened_text_vec: List[int] = []
    #     for vec in text_vecs:
    #         for token in vec:
    #             flattened_text_vec.append(token)

    #     # Format history vec given various logic
    #     if self.text_truncate is not None:
    #         truncate_length = self.text_truncate - 2
    #         if len(flattened_text_vec) > truncate_length:
    #             flattened_text_vec = flattened_text_vec[-truncate_length:]
    #     flattened_text_vec = torch.tensor(flattened_text_vec, dtype=torch.long)
    #     # originally "if is_bart: Seems to be excluded in Emely"
    #     # flattened_text_vec = torch.cat(
    #     #     [
    #     #         torch.tensor([self.start_idx], dtype=torch.long),
    #     #         flattened_text_vec,
    #     #         torch.tensor([self.end_idx], dtype=torch.long),
    #     #     ],
    #     #     dim=0,
    #     # )

    #     # Pass through the encoder and decoder to generate tokens
    #     batch_text_vec = torch.unsqueeze(flattened_text_vec, dim=0)  # Add batch dim
    #     encoder_states = self.encoder(batch_text_vec)
    #     generations = self._get_initial_decoder_input(batch_text_vec)
    #     # keep track of early stopping if all generations finish
    #     seen_end = torch.zeros(
    #         batch_text_vec.size(0), device=batch_text_vec.device, dtype=torch.bool
    #     )
    #     incr_state: Dict[str, torch.Tensor] = {}
    #     for token_idx in range(max_len):
    #         if token_idx == 0:
    #             latent, incr_state = self.decoder_first_pass(
    #                 generations, encoder_states
    #             )
    #         else:
    #             latent, incr_state = self.decoder_later_pass(
    #                 generations, encoder_states, incr_state
    #             )
    #         logits = self.partially_traced_model.output(latent[:, -1:, :])
    #         _, preds = logits.max(dim=2)
    #         incr_state = self.partially_traced_model.reorder_decoder_incremental_state(
    #             incr_state,
    #             torch.tensor([0], dtype=torch.long, device=batch_text_vec.device),
    #         )
    #         seen_end = seen_end + (preds == self.end_idx).squeeze(1)
    #         generations = torch.cat([generations, preds], dim=1)
    #         if torch.all(seen_end):
    #             break

    #     # Get the label from the generated tokens and update the history
    #     generation_tokens: List[int] = generations[0].tolist()
    #     label = self._v2t(generation_tokens)

    #     return label


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
        dict_lower: bool,
    ):

        self.null_token = null_token
        self.end_token = end_token
        self.unk_token = unk_token
        self.start_token = start_token

        self.dict_lower = dict_lower

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
        if self.dict_lower:
            text = text.lower()
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

@torch.jit.script
class _ScriptableHypothesisTail(object):
    """
    Hold some bookkeeping about a hypothesis.
    """

    # use slots because we don't want dynamic attributes here
    __slots__ = ['timestep', 'hypid', 'score', 'tokenid']

    def __init__(self, timestep: int, hypid: int, score: float, tokenid: int):
        self.timestep = timestep
        self.hypid = hypid
        self.score = score
        self.tokenid = tokenid

@torch.jit.script
def scriptedNeginf(dtype: torch.dtype) -> float:
    """
    Return a representable finite number near -inf for a dtype.
    """
    if dtype is torch.float16:
        return -65504.0
    else:
        return -10.0**20

@torch.jit.script
class ScriptableTreeSearch(object):
    """
    Abstract Tree Search class.

    It keeps information about beam_size concurrent, developing hypotheses. Concrete
    implementations make choices about which token to explore next at each point in the
    tree. Different choices result in different generation algorithms.
    """

    def __init__(
        self,
        beam_size: int,
        inference: str,
        block_ngram: int = -1,
        context_block_ngram: int = -1,
        padding_token: int = 0,
        bos_token: int = 1,
        eos_token: int = 2,
        min_length: int = 3,
        length_penalty: float = 0.65,
    ):
        """
        Instantiate Beam object.

        :param beam_size:
            number of hypothesis in the beam
        :param block_ngram:
            size of ngrams to block.
        :param context_block_ngram:
            size of context ngrams to block
        :param padding_token:
            padding token ID
        :param bos_token:
            beginning of sentence token ID
        :param eos_token:
            end of sentence token ID
        :param min_length:
            minimum length of the predicted sequence
        :param device:
            What device to use for computations
        """
        self.inference = inference
        self.beam_size = beam_size
        self.length_penalty = length_penalty
        self.block_ngram = block_ngram
        self.min_length = min_length
        self.eos = eos_token
        self.bos = bos_token
        self.pad = padding_token
        self.context_block_ngram = context_block_ngram
        #self.block_list: Optional[SearchBlocklist] = None # Not currently supported by this implementation
        # recent score for each hypo in the beam
        self.scores: Optional[torch.Tensor] = torch.zeros(1)
        # self.scores values per each time step
        self.all_scores = [torch.tensor([0.0] * beam_size)]
        # backtracking id to hypothesis at previous time step
        self.bookkeep = []
        # output tokens at each time step
        self.outputs = [
            torch.tensor([self.beam_size]).long().fill_(self.bos)
        ]
        # keeps tuples (score, time_step, hyp_id)
        self.finished: List[_ScriptableHypothesisTail] = []
        self.eos_top = False
        self.eos_top_ts: int = -1
        self.n_best_counter = 0
        self.partial_hyps: List[List[int]] = [[self.bos] for i in range(beam_size)]

        self.context: List[int] = []

    def set_context(self, context: torch.LongTensor):
        """
        Set the internal context representation and return self.

        :param context:
            a LongTensor representing the input context; used for context
            ngram blocking, if supplied
        """
        self.context: List[int] = context.tolist()

    # def set_block_list(self: TSType, block_list: Optional[SearchBlocklist]) -> TSType:
    #     self.block_list = block_list
    #     return self

    def get_output_from_current_step(self) -> torch.Tensor:
        """
        Get the outputput at the current step.
        """
        return self.outputs[-1]

    def get_backtrack_from_current_step(self) -> torch.Tensor:
        """
        Get the backtrack at the current step.
        """
        return self.bookkeep[-1]

    def select_paths(self, logprobs: torch.Tensor, prior_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select the next vocabulary item in these beams.

        :param logprobs:
            a (beamsize x vocab) tensor of log probabilities. If this is the first
            turn in the dialogue, it will be a (1 x vocab) tensor.
        :param prior_scores:
            a (beamsize) tensor of weights with the cumulative running
            log-probability of each beam. If the first turn, it will be a (1) tensor.
        :param current_length:
            the current length in tokens
        :return:
            a (hypothesis_ids, token_id, scores) tuple, where:

            - hypothesis_ids is a LongTensor of hypotheses we're extending. May have
              repeats, but should always be (beamsize) long.
            - token_ids is a (beamsize) LongTensor of next-token choices for
              each of the hypotheses.
            - scores is a (beamsize) Tensor with the updated cumulative log-probs
              of each beam.
        """
        if self.inference == "beam":
            """ BEAM SEARCH """
            # if numel is 1, then this is the first time step, only one hyp is expanded
            if prior_scores.numel() == 1:
                logprobs = logprobs[0:1]

            # beam search actually looks over all hypotheses together so we flatten
            beam_scores = logprobs + prior_scores.unsqueeze(1).expand_as(logprobs)
            flat_beam_scores = beam_scores.view(-1)
            best_scores, best_idxs = torch.topk(flat_beam_scores, self.beam_size, dim=-1)
            voc_size = logprobs.size(-1)

            # get the backtracking hypothesis id as a multiple of full voc_sizes
            hyp_ids = best_idxs // voc_size
            # get the actual word id from residual of the same division
            tok_ids = best_idxs % voc_size

            return (hyp_ids, tok_ids, best_scores)
        """ GREEDY SEARCH"""
        if self.beam_size != 1:
            raise ValueError('Greedy search can only be run with beam size 1.')
        tok_scores, tok_ids = logprobs.max(1)
        best_scores = tok_scores + prior_scores
        hyp_ids = torch.arange(logprobs.size(0))
        return (hyp_ids, tok_ids, best_scores)

    def _block_ngrams(
        self, ngram_size: int, logprobs: torch.Tensor, source: Optional[List[int]]) -> torch.Tensor:
        """
        Hard block ngrams from the logprobs, based on the source.

        :param ngram_size:
            The length of ngrams to block. Must be > 0.
        :param logprobs:
            Float or HalfTensor, representing the log-probabilities. This is
            modified in place.
        :param source:
            Source text to grab ngrams from. If None, it uses the current
            hypothesis (i.e. self-blocking).
        """
        for beam_id, hyp in enumerate(self.partial_hyps):
            if len(hyp) < ngram_size - 1:
                continue
            source_ = hyp if source is None else source
            ngrams = [source_[i:i+ngram_size] for i in range(len(source_)-ngram_size+1)]
            prefix = hyp[-(ngram_size - 1) :]
            for ngram in ngrams:
                if ngram_size == 1 or prefix == ngram[:-1]:
                    logprobs[beam_id][ngram[-1]] = scriptedNeginf(logprobs.dtype)
        return logprobs

    # def _block_block_list(self, logprobs: torch.Tensor) -> torch.Tensor:
    #     if self.block_list is None:
    #         return logprobs

    #     for beam_id, hyp in enumerate(self.partial_hyps):
    #         for ngram_size, bad_ngrams in self.block_list.items():
    #             prefix = hyp[-(ngram_size - 1) :]
    #             for ngram in bad_ngrams:
    #                 if (ngram_size == 1) or prefix == list(ngram[:-1]):
    #                     logprobs[beam_id][ngram[-1]] = neginf(logprobs.dtype)
    #     return logprobs

    def advance(self, logprobs: torch.Tensor):
        """
        Advance the beam one step.
        """
        current_length = len(self.all_scores) - 1
        if current_length < self.min_length:
            # penalize all eos probs to make it decode longer
            for hyp_id in range(logprobs.size(0)):
                logprobs[hyp_id][self.eos] = scriptedNeginf(logprobs.dtype)

        self.scores = self.scores.type_as(logprobs)

        # penalize hypotheses ending in EOS on the prior scores (self.scores) level
        # this is related to search which uses prior scores (self.scores) (e.g. beam)
        for hyp_id, token in enumerate(self.outputs[-1]):
            if token == self.eos:
                self.scores[hyp_id] = scriptedNeginf(self.scores.dtype)

        # beam blocking
        if self.block_ngram > 0:
            logprobs = self._block_ngrams(self.block_ngram, logprobs, None)

        #logprobs = self._block_block_list(logprobs)

        if self.context_block_ngram > 0:
            if self.context is None:
                raise ValueError(
                    "Must use TreeSearch.set_context to use context blocking."
                )
            logprobs = self._block_ngrams(
                self.context_block_ngram, logprobs, self.context
            )

        hyp_ids, tok_ids, self.scores = self.select_paths(
            logprobs, self.scores
        )
        # use clone() here to ensure that self.all_scores will not be changed
        # later due to any penalties to self.scores
        self.all_scores.append(self.scores.clone())

        self.outputs.append(tok_ids)
        self.bookkeep.append(hyp_ids)
        temphyps: List[List[int]] = []
        for i in range(self.beam_size):
            temp: List[int] = []
            tempvals = self.partial_hyps[hyp_ids[i]]
            for val in tempvals:
                temp.append(val)
            temp.append(tok_ids[i].item())
            temphyps.append(temp)
        self.partial_hyps = temphyps

        #  check new hypos for eos label, if we have some, add to finished
        for hypid in range(self.beam_size):
            if self.outputs[-1][hypid] == self.eos:
                if self.scores[hypid] <= scriptedNeginf(self.scores.dtype):
                    continue
                #  this is finished hypo, adding to finished
                eostail = _ScriptableHypothesisTail(
                    timestep=len(self.outputs) - 1,
                    hypid=hypid,
                    score=self.all_scores[-1][hypid].item(),
                    tokenid=self.eos,
                )
                self.finished.append(eostail)
                self.n_best_counter += 1

        if self.outputs[-1][0] == self.eos:
            self.eos_top = True
            if self.eos_top_ts==-1:
                self.eos_top_ts = len(self.outputs) - 1

    def is_done(self) -> bool:
        """
        Return whether beam search is complete.
        """
        return self.eos_top and self.n_best_counter >= self.beam_size

    # def _find_ngrams(self, input_list, n):
    #     """
    #     Find ngrams of size n in input list.
    #     """
    #     return list(zip(*[input_list[i:] for i in range(n)]))

    def _get_hyp_from_finished(self, hypothesis_tail: _ScriptableHypothesisTail) -> List[_ScriptableHypothesisTail]:
        hyp_idx: List[_ScriptableHypothesisTail] = []
        endback: int = hypothesis_tail.hypid
        for i in range(hypothesis_tail.timestep, -1, -1):
            hyp_idx.append(
                _ScriptableHypothesisTail(
                    timestep=i,
                    hypid=endback,
                    score=self.all_scores[i][endback].item(),
                    tokenid=self.outputs[i][endback].item(),
                )
            )
            endback = self.bookkeep[i - 1][endback]

        return hyp_idx

    def _get_pretty_hypothesis(self, list_of_hypotails: List[_ScriptableHypothesisTail]) -> torch.Tensor:
        """
        Return hypothesis as a tensor of token ids.
        """
        reslist: torch.Tensor = torch.zeros(len(list_of_hypotails))
        for i in range(len(list_of_hypotails)):
            reslist[i] = list_of_hypotails[len(list_of_hypotails)-i-1].tokenid
        return reslist

    def get_rescored_finished(self) -> torch.Tensor:
        """
        Return finished hypotheses according to adjusted scores.

        Score adjustment is done according to the Google NMT paper, which
        penalizes long utterances.

        :param n_best:
            number of finalized hypotheses to return

        :return:
            list of (tokens, score) pairs, in sorted order, where:
              - tokens is a tensor of token ids
              - score is the adjusted log probability of the entire utterance
        """
        # if we never actually finished, force one
        if not self.finished:
            self.outputs[-1][0] = self.eos
            self.finished.append(
                _ScriptableHypothesisTail(
                    timestep=len(self.outputs) - 1,
                    hypid=0,
                    score=self.all_scores[-1][0].item(),
                    tokenid=self.outputs[-1][0].item(),
                )
            )

        scores: torch.Tensor = torch.zeros(len(self.finished))
        rescored_finished: List[_ScriptableHypothesisTail] = []
        for i,finished_item in enumerate(self.finished):
            current_length = finished_item.timestep + 1
            # these weights are from Google NMT paper
            length_penalty = ((1 + current_length) / 6)**self.length_penalty
            rescored_finished.append(
                _ScriptableHypothesisTail(
                    timestep=finished_item.timestep,
                    hypid=finished_item.hypid,
                    score=finished_item.score / length_penalty,
                    tokenid=finished_item.tokenid,
                )
            )
            scores[i] = finished_item.score / length_penalty
        
        maxscore_ind = torch.argmax(scores).item()

        best: torch.Tensor = self._get_pretty_hypothesis(self._get_hyp_from_finished(rescored_finished[maxscore_ind]))
        
        return best