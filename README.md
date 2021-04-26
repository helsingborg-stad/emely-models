## The models used by EmelyBackend

### Setup instructions for training

1. cd into ParlAI and run $python setup.py develop
2. cd into huggingface and run $python pip install -r requirements.txt


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
├── huggingface        <- Everything huggingface related: training, deployment, api
