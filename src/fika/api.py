from typing import Dict
from fastapi import FastAPI, Response, status
from pydantic import BaseModel
from pathlib import Path

from parlai.agents.emely.emely import EmelyAgent
from parlai.core.opt import Opt
import logging


class ApiMessage(BaseModel):
    text: str

class NewOpt(BaseModel):
    new_opts: Dict


app = FastAPI()

# Logging is used in Emely
logging.basicConfig(level=logging.NOTSET)

# Opts for model loading
model_path = Path(__file__).resolve().parents[2] / 'models/blender_400Mdistill'
opt_path = model_path / 'model.opt'
opt = Opt.load(opt_path.as_posix())
# opt['task'] = 'internal'
opt['skip_generation'] = False
opt['init_model'] = (model_path / 'model').as_posix()
opt['no_cuda'] = True  # Cloud run doesn't offer gpu support


model: EmelyAgent


@app.on_event("startup")
async def startup_event():
    global model
    model = EmelyAgent(opt)
    return


@app.post("/inference")
async def inference(msg: ApiMessage):
    # Async model throughput
    reply = model.observe_and_act(msg.text)

    return ApiMessage(text=reply)


# This endpoint can be used to both wake up the program and get the name of the model
@app.get('/model-name')
def get_model():
    model_name = model_path.name
    return ApiMessage(text=model_name)


@app.post("/opt")
def change_opt(opts: NewOpt, response: Response):

    opt_dict = opts.new_opts
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

    global model
    try:
        model = EmelyAgent(opt)
        message = create_success_message(param_changes)
        return ApiMessage(text=message)

    except Exception as e:
        response.status_code = status.HTTP_500
        return ApiMessage(text=f'Error: {e}')
    

def create_success_message(param_changes):
    text = 'Sucess!'
    for k, v in param_changes.items():
        text = text + f'\n{k}: {v[0]} -> {v[1]}'
    return text
        
        