from fastapi import FastAPI
import joblib
from pydantic import BaseModel
from io import StringIO
import xgboost as xgb
import pandas as pd
import numpy as np
import json
import sys
import os

print("\n".join(sys.path))
from ml.model import create_prediction_attributes, inference

model_path = os.path.join(os.path.dirname(__file__), "model/xgb.model")

print('Loading model from: ', model_path)

# load model from file[]
model = xgb.Booster()
model.load_model(model_path)

# Define the FastAPI app
app = FastAPI()

# Define the input data model for prediction
class InputFeatures(BaseModel):
    age: float
    workclass: str
    fnlgt: float
    education: str
    education_num: float
    marital_status: str
    occupation:str
    relationship: str
    race: str
    sex: str
    capital_gain: float
    capital_loss: float
    hours_per_week: float
    native_country: str
    

# Define the endpoint for prediction using PUT method
@app.put("/predict")
def predict(input_data: InputFeatures):
    """
    Predicts the salary class of a given input data.
    """
    # convert input data to dataframe
    print( "Input data: ", input_data)

    # replace _ with - in the column names
    input_data = input_data.dict()
    input_data = {k.replace("_", "-"): v for k, v in input_data.items()}
    
    input_df = pd.DataFrame([input_data])
    
    encoder = joblib.load(os.path.join(os.path.dirname(__file__), "model/encoder.joblib"))
    
    X = create_prediction_attributes(encoder, input_df)
       
    # make predictions
    preds = inference(model, X)
    
    # return predictions
    return {"prediction": preds[0]}


