from typing import Dict, List
from fastapi import FastAPI, Response, status, BackgroundTasks
from pydantic import BaseModel
from pathlib import Path

from src.emely.agent import EmelyAgent
from parlai.core.opt import Opt
import logging
import os


class ApiMessage(BaseModel):
    text: str
    block_list: List[str] = None


class ModelOpt(BaseModel):
    opts: Dict


app = FastAPI()
logging.basicConfig(level=logging.NOTSET)

# Opts for model loading
model_path = Path(__file__).resolve().parent / "models" / os.environ.get("MODEL_NAME", 'blender_90M')
opt_path = model_path / "model.opt"
opt = Opt.load(opt_path.as_posix())
opt["skip_generation"] = False
opt["init_model"] = (model_path / "model").as_posix()
opt["no_cuda"] = True  # Cloud run doesn't offer GPU support

# Inference options
opt["beam_context_block_ngram"] = 3
opt["inference"] = "beam"
opt["beam_size"] = 10
opt["beam_min_length"] = 15

model: EmelyAgent


@app.on_event("startup")
async def startup_event():
    global model
    model = EmelyAgent(opt)
    return


@app.post("/inference")
async def inference(msg: ApiMessage):
    if msg.block_list is None:
        msg.block_list = []

    # Sets beam_block_list
    if len(msg.block_list) > 0:
        model.set_block_list(block_list=msg.block_list)

    reply = model.observe_and_act(msg.text)
    return ApiMessage(text=reply)


# This endpoint can be used to both wake up the program and get the name of the model
@app.get("/wake")
async def get_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(startup_event)
    model_name = opt["model_file"]
    return ApiMessage(text=model_name)

