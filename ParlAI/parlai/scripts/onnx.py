#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch.jit
import torch.nn as nn
from packaging import version

from parlai.core.agents import create_agent
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript, register_script
from parlai.utils.io import PathManager
from parlai.agents.emely.emely import EmelyAgent
from parlai.utils.bpe import SubwordBPEHelper

def export_emely_onnx(opt: Opt):
    """
    Export Emely to TorchScript so that inference can be run outside of ParlAI.
    """

    if version.parse(torch.__version__) < version.parse('1.7.0'):
        raise NotImplementedError(
            'TorchScript export is only supported for Torch 1.7 and higher!'
        )
    else:
        from parlai.torchscript.modules_emely_onnx import ONNXEmelyAgent

    overrides = {
        'no_cuda': True,  # TorchScripting is CPU only
        'model_parallel': False,  # model_parallel is not currently supported when TorchScripting
    }
    if 'override' not in opt:
        opt['override'] = {}
    for k, v in overrides.items():
        opt[k] = v
        opt['override'][k] = v

    # Create the unscripted greedy-search module
    agent = EmelyAgent(opt)
    sbpe = SubwordBPEHelper(agent.opt)
    joint_bpe_codes = {}
    for k in sbpe.bpe.bpe_codes.keys():
        joint_bpe_codes[agent.opt["temp_separator"].join(k)] = sbpe.bpe.bpe_codes[k]
    sbpe.bpe_codes = joint_bpe_codes
    sbpe.separator = "@@"
    agent.dict.bpe = sbpe
    original_module = ONNXEmelyAgent(agent)

    # # Script the module and save
    # scripted_module = torch.jit.script(TorchScriptedEmelyAgent(agent))
    # with PathManager.open(opt['scripted_model_file'], 'wb') as f:
    #     torch.jit.save(scripted_module, f)

    # # Compare the original module to the scripted module against the test inputs
    # if len(opt['input']) > 0:
    #     inputs = opt['input'].split('|')
    #     print('\nGenerating given the original unscripted module:')
    #     _run_conversation(module=original_module, inputs=inputs)
    #     print('\nGenerating given the scripted module:')
    #     _run_conversation(module=scripted_module, inputs=inputs)
    
    return original_module#, scripted_module


def _run_conversation(module: nn.Module, inputs: List[str]):
    """
    Run a conversation with the given module given the input strings.
    """
    context = []
    for input_ in inputs:
        print(' TEXT: ' + input_)
        context.append(input_)
        label = module('\n'.join(context))
        print("LABEL: " + label)
        context.append(label)
