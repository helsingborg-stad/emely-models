The models used by EmelyBackend

# Overview
This project was developed by NordAxon for Helsingborg Stad 2021-2022, and open sourced in April 2023.

Emely is a Conversational AI System for practicing job interviews in Swedish. The dialogue model is a finetuned Blenderbot 1.

The project is separated in 3 repositories: [emely-frontend](https://github.com/helsingborg-stad/emely-frontend), [emely-backend](https://github.com/helsingborg-stad/emely-backend) & [emely-models](https://github.com/helsingborg-stad/emely-models).

## Installing the repo locally

Make sure you're on Linux or WSL
 ´´´sh install.sh´´´´

## Deploying a model!

sh deploy.sh -m MODEL_NAME -s SERVICE

where MODEL_NAME is the name of a directory under models/
SERVICE is either fika or interview


## Project structure
```
├── LICENSE
├── README.md          <- The top-level README for developers using this project.
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment
|
├── ParlAI             <- ParlAI source code
    ├── setup.py       <- setup file used to install parlai and it's dependencies properly
|
├── src                <- Repo code
```
Files: 
- app.py - FastAPI app to run an Emely model. Requires a environment variable MODEL_NAME to point to the model directory
- deploy.sh - Use to deploy a new fika/interview model
- gcloudignore-standard.txt - Standard gcloudignore that is used during deploy.sh
- install.sh
- setup.py - install src as python package using pip install -e .
