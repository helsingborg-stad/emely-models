## The models used by EmelyBackend

### Setup instructions for training

1. cd into ParlAI and run $python setup.py develop


### How to deploy
Unfortunately this is a bit difficult. Why? We only want one model to go to GCP with a deployment and we have to specify it in several places:
1. In the variable model_path src/< persona >/api.py file
    - model_path = Path(__file__).resolve().parents[2] / 'models/MODEL_DIR'
2. In the Dockfile where we copy it over to the image: src/< persona >/Dockerfile 
    - COPY models/MODEL_DIR ./models/MODEL_DIR
3. Make sure it's *NOT* in the .gitignore file in root folder
    - Remove it from .gitignore
    - Add all other models to the .gitignore

The last step is necessary because when deploying, google will automatically ignore what's in the .gitignore

The deployment commands are (replace the SERVICE): 
$ gcloud builds submit --tag eu.gcr.io/emelybrainapi/< SERVICE > --timeout 3000s && gcloud run deploy --image eu.gcr.io/emelybrainapi/< SERVICE > --platform managed --memory 2Gi --region europe-west1 --allow-unauthenticated --timeout 900s

- Replace the SERVICE
- SERVICE cannot be uppercase



### Project structure
├── LICENSE
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── json           <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
││
├── requirements.txt   <- The requirements file for reproducing the analysis environment
|
├── ParlAI             <- ParlAI source code
    ├── setup.py       <- setup file used to install parlai and it's dependencies properly
|
├── parlai-src         <- Dir for using parlai: training, deployment, api etc
|
├── models                 <- Model files
    ├── fika-model         <- fika deployment
    ├── interview-model    <- interview deployment
    ├── runs               <- runs from training

