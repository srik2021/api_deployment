This project illustrates some of the best practices of MLOps - including developing modularized code, writing unit tests, testing performance of models over slices of data and implementing CI/CD

# Environment Set up
The requirements.txt file includes the libraries required to train models, execute unit tests and deploy the model as a fastAPI on heroku

# Repositories
The repository is hosted at https://github.com/srik2021/api_deployment

# Directory structure
starter is the primary directory containing code, data, model, logs, screenshots and tests.  

## Source
The source code is under the 'src' directory

## Data
The 'data' directory contains the raw census.csv used to train and validate the model.  It also stores sample input (raw and transformed) and corresponding predictions generated during validation and is used in tests to ensure that the same predications are generated

## Model
The 'model' folder contains the trained model as well as encoder.  It is used as part of unit tests to compare against the saved sample input and predictions.  They are also loaded by fastAPI for transformations and predictions on input

## Tests
The 'tests' directory contains pytest to both test the transformation and training code as well as test the fastAPI get and post methods

# API Creation
*  Created a RESTful API using FastAPI that implements:
    * GET on the root giving a welcome message.
    * POST that does model inference.
    * Uses type hinting and pydantic model to ingest the body from POST. T

# API Deployment
* The model API has been deployed on heroku and is enabled for Continuous Integration(CI)
* A new build and deployment is automatically triggered by new commits to the main branch to the github code repository.

