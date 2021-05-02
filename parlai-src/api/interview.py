from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
from pathlib import Path

from parlai.agents.emely.emely import EmelyAgent
from parlai.core.opt import Opt
import logging


class ApiMessage(BaseModel):
    text: str


app = FastAPI()

# Logging is used in Emely
logging.basicConfig(level=logging.NOTSET)

# Opts for model loading
model_path = Path(__file__).resolve().parents[2] / 'deploy-model'
opt_path = model_path / 'model.opt'
opt = Opt.load(opt_path.as_posix())
opt['task'] = 'internal'
opt['skip_generation'] = False
opt['init_model'] = (model_path / 'model').as_posix()
opt['no_cuda'] = True # Don't assume cuda on the gcp instance
# TODO: DO I need anything else here?

#
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

# TODO
@app.get('/model-name')
def get_model():
    raise NotImplementedError()
