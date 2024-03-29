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

    class Config:
        alias_generator = lambda x: x.replace("-", "_")
        allow_population_by_field_name = True
        
        schema_extra = {
            "example": {
                "age": 26,
                "workclass": "Private",
                "fnlgt": 132661,
                "education": "HS-grad",
                "education_num": 9,
                "marital_status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Wife",
                "race": "White",
                "sex": "Female",
                "capital_gain": 5013,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States"
            }
        }
        
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

# Define the endpoint for prediction using PUT method
@app.post("/predict_salary")
def predict_salary(input_data: InputFeatures):
    """
    Endpoint for making predictions.
    """
    print('Predicting salary for input data: ', input_data)
    return predict(input_data)

# Define the endpoint for prediction using GET method
@app.get("/get_salary_prediction")    
def get_salary_prediction():
    """
    Endpoint for providing information about api.
    """
    # Read contents of welcome.txt to welcome_txt
    with open("starter/welcome.txt", 'r') as f:
        welcome_txt = f.read()
    
    return welcome_txt

    
    
 


