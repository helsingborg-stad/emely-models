from parlai.core.agents import create_agent
from parlai.agents.emely.emely import EmelyAgent
from parlai.core.opt import Opt
from pathlib import Path
import torch
from parlai.scripts.torchscript import export_emely
from parlai.core.agents import create_agent

# Initialize model settings

model_path = Path.cwd() / 'data/models/blender/blender_90M/'
assert model_path.is_dir()

opt_path = model_path / 'model.opt'
opt = Opt.load(opt_path)

# Change opts
opt['skip_generation'] = False
opt['init_model'] = (model_path / 'model').as_posix()
opt['no_cuda'] = True  # Cloud run doesn't offer gpu support

# Inference options
opt['inference'] = 'beam' # 'beam'
opt['beam_size'] = 10

opt["scripted_model_file"] = "../../saved_models/emely_scripted_test.pt"
opt["script-module"] = "parlai.torchscript.modules:TorchScriptGreedySearch"
opt["model_file"] = opt["init_model"]

opt["temp_separator"] = "__space__"

opt["bpe_add_prefix_space"] = False

opt["input"] = "Hi Emely, how are you?\nI'm good thanks! What do you do for work?\nI write code and I drink coffe."

original_module, scripted_module = export_emely(opt)

text = "Hi Emely, how are you?\nI'm good thanks! What do you do for work?\nI write code and I drink coffe."
reply = original_module(text)