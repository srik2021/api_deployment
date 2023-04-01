import json
from fastapi import FastAPI
from fastapi.testclient import TestClient
from urllib.parse import unquote
import joblib
import pandas as pd
import pytest

from main import app

# fixture to create fastAPI client
@pytest.fixture
def client():
   return TestClient(app)


def test_model_api_post(client):
    """
    Tests that the model API returns the correct predictions.
    """
    # load sample input and predictions from file
    sample_df = pd.read_csv("starter/data/sample_raw_input_and_predictions_name.csv")
    
    # get sample input and predictions
    X_test = sample_df.drop("predictions", axis=1)
    saved_predictions = sample_df["predictions"]
    
    # make predictions using API
    predictions = []
    for _, row in X_test.iterrows():
        print('Raw input: ', row.to_dict())
        response = client.post("/predict_salary", json=row.to_dict())
        assert response.status_code == 200
        predictions.append(response.json().get('prediction'))
        print('Predictions: ', response.json())
        
    # test that predictions are the same as saved predictions
    assert predictions == saved_predictions.tolist()

def test_model_api_get(client):
    """
    Tests that the api returns welcome contents on get invocation
    """

    response = client.get("/get_salary_prediction")   
    assert response.status_code == 200
    response_text = response.text
   
    #assert api returns contents of welcome.txt
    with open("starter/welcome.txt", 'r', encoding='utf-8') as file:
        welcome_txt = file.read()
        
    # assert that the welcome_txt is in the response text.  
    # Using in instead of == because of extra quotes in response
    assert welcome_txt in response_text



