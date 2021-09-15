from parlai.agents.emely.emely import EmelyAgent
from parlai.core.opt import Opt
from pathlib import Path
import time

model_path = Path.cwd() / 'ParlAI/data/models/blender/blender_90M/'
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

def test_time(n,emely_agent):
    s = 'HI emely, this is a test to see if you are any faster now than you used to be haha'
    start = time.time()

    for _ in range(n):
        emely_agent.observe_and_act(s)

    average = (time.time() - start) / n 

    print(f'Average time was {average}')

test_time(50,emely_agent)