from parlai.agents.emely.emely import EmelyAgent
from parlai.core.opt import Opt
from pathlib import Path
import time
import torch

model_path = Path.cwd() / 'models/blender_90M/'
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

emely_agent = EmelyAgent(opt)
emely_agent.model = torch.quantization.quantize_dynamic(emely_agent.model, {torch.nn.Linear}, dtype=torch.qint8) 

def test_time(n,emely_agent):
    s = 'Hi'
    start = time.time()

    for _ in range(n):
        reply = emely_agent.observe_and_act(s)

    average = (time.time() - start) / n 

    print(f'Average time was {average}')
    return reply

reply = test_time(1,emely_agent)
print(reply)