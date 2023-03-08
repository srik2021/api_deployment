from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import json
import sys
import os

print("\n".join(sys.path))
from src.ml.model import create_prediction_attributes, inference

model_path = os.path.join(os.path.dirname(__file__), "model/xgb.model")

print('Loading model from: ', model_path)

# load model from file
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
    native_country: float
    

# Define the endpoint for prediction using PUT method
@app.put("/predict")
def predict(input_data: InputFeatures):
    """
    Predicts the salary class of a given input data.
    """
    # convert input data to dataframe
    input_df = pd.DataFrame([input_data.dict()])
    
    X = create_prediction_attributes(input_df)
    
    # make predictions
    preds = inference(model, X)
    
    # return predictions
    return {"prediction": preds[0]}


