from typing import Dict
from fastapi import FastAPI, Response, status, BackgroundTasks
from pydantic import BaseModel
from pathlib import Path

from parlai.agents.emely.emely import EmelyAgent
from parlai.core.opt import Opt
import logging
import torch


class ApiMessage(BaseModel):
    text: str

class ModelOpt(BaseModel):
    opts: Dict


app = FastAPI()

# Logging is used in Emely
logging.basicConfig(level=logging.NOTSET)

# Opts for model loading
model_path = Path(__file__).resolve().parents[2] / 'models/blender_90M'
opt_path = model_path / 'model.opt'
opt = Opt.load(opt_path.as_posix())

# Change opts 
opt['skip_generation'] = False
opt['init_model'] = (model_path / 'model').as_posix()
opt['no_cuda'] = True  # Cloud run doesn't offer gpu support

# Inference options
opt['inference'] = 'topk'
opt['beam_size'] = 10
opt['topk'] = 40


model: EmelyAgent


@app.on_event("startup")
async def startup_event():
    " Loads model and quantizes it with torch"
    global model, opt
    model = EmelyAgent(opt)
    model.model = torch.quantization.quantize_dynamic(model.model, {torch.nn.Linear}, dtype=torch.qint8)
    return


@app.post("/inference")
async def inference(msg: ApiMessage):
    reply = model.observe_and_act(msg.text)
    return ApiMessage(text=reply)


# This endpoint can be used to both wake up the program and get the name of the model
@app.get('/model-name')
async def get_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(startup_event)
    model_name = opt["model_file"]
    return ApiMessage(text=model_name)


@app.get("/opt")
async def get_opt():
    options = ["inference", "beam_size", "beam_min_length", "topp", "topk"]
    inference_opt = {opt: model.opt[opt] for opt in options}
    return ModelOpt(opts=inference_opt)


@app.post("/opt")
async def change_opt(new_opts: ModelOpt, response: Response, background_tasks: BackgroundTasks):

    opt_dict = new_opts.opts
    global opt
    param_changes = {}

    for k, v in opt_dict.items():
        try:
            old_value = opt[k]
            assert type(v) is type(old_value)
            opt[k] = v
            param_changes[k] = (old_value, v)
        except KeyError as e:
            response.status_code = status.HTTP_400_BAD_REQUEST
            return ApiMessage(text=f'Keyerror: {e}')
        except AssertionError:
            response.status_code = status.HTTP_400_BAD_REQUEST
            return ApiMessage(text=f'New value for {k} was {type(v)} but should be {type(old_value)}')

    try:
        # Reload model with new opts     
        background_tasks.add_task(startup_event)
        message = create_success_message(param_changes)
        return ApiMessage(text=message)

    except Exception as e:
        response.status_code = status.HTTP_500
        return ApiMessage(text=f'Error: {e}')
    

def create_success_message(param_changes):
    "Creates message information what opts were updated when using POST @/opt"
    text = 'Sucess!'
    for k, v in param_changes.items():
        text = text + f'\n{k}: {v[0]} -> {v[1]}'
    return text
        