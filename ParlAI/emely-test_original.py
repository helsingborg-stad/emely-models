from parlai.core.agents import create_agent
from parlai.agents.emely.emely import EmelyAgent
from parlai.core.opt import Opt
from pathlib import Path
import torch

model_path = Path.cwd() / 'data/models/blender/blender_90M/'
assert model_path.is_dir()

opt_path = model_path / 'model.opt'
opt = Opt.load(opt_path)

# Change opts 
opt['skip_generation'] = False
opt['init_model'] = (model_path / 'model').as_posix()
opt['no_cuda'] = True  # Cloud run doesn't offer gpu support

# Inference options
opt['inference'] = 'greedy' # 'beam'
opt['beam_size'] = 1

emely_agent = EmelyAgent(opt)

text = "Hi Emely, how are you?\nI'm good thanks! What do you do for work?\nI write code and I drink coffe"
emely_agent.observe_and_act(text)