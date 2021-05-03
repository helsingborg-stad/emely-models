from fastapi import FastAPI
from transformers import BlenderbotSmallForConditionalGeneration, BlenderbotSmallTokenizer
from pathlib import Path
from pydantic import BaseModel
import torch
import asyncio


class ApiMessage(BaseModel):
    text: str


app = FastAPI()

model_name = 'huggingface-models/blenderbot_small-90M'
model_path = Path(__file__).resolve().parents[2] / model_name
model: BlenderbotSmallForConditionalGeneration
tokenizer: BlenderbotSmallTokenizer
device: str


@app.on_event("startup")
async def startup_event():
    """ Sets the model, tokenizer and device on app startup """
    global model, tokenizer, device
    model = BlenderbotSmallForConditionalGeneration.from_pretrained(model_path / 'model')
    tokenizer = BlenderbotSmallTokenizer.from_pretrained(model_path / 'tokenizer')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return


@app.post('/inference')
async def inference(msg: ApiMessage):
    loop = asyncio.get_event_loop()

    # Async model throughput
    reply = await loop.run_in_executor(None, model_inference, msg.text)

    return ApiMessage(text=reply)


def model_inference(text):
    """ Function used at inference to enable run_in_executor """
    context = text
    inputs = tokenizer([context], return_tensors='pt')
    inputs.to(device)
    with torch.no_grad():
        output_tokens = model.generate(**inputs)  # Using the default executor
    reply = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return reply

# This endpoint can be used to both wake up the program and get the name of the model
@app.get('/model-name')
def get_model():
    return ApiMessage(text=model_name)

