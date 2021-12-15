## The models used by EmelyBackend

# WAndb api key
fc7bf1f22555377e0a2d337667eadd6547da5c97

### Installing the repo locally

Make sure your on Linux or WSL
 ´´´sh install.sh´´´´

### Deploying a model!

sh deploy.sh -m MODEL_NAME -s SERVICE

where MODEL_NAME is the name of a directory under models/
SERVICE is either fika or interview


### Project structure
├── LICENSE
├── README.md          <- The top-level README for developers using this project.
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment
|
├── ParlAI             <- ParlAI source code
    ├── setup.py       <- setup file used to install parlai and it's dependencies properly
|
├── src                <- Repo code

Files: 
- app.py - FastAPI app to run an Emely model. Requires a environment variable MODEL_NAME to point to the model directory
- deploy.sh - Use to deploy a new fika/interview model
- gcloudignore-standard.txt - Standard gcloudignore that is used during deploy.sh
- install.sh
- setup.py - install src as python package using pip install -e .
