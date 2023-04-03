import json
import sys
from fastapi import FastAPI
from fastapi.testclient import TestClient
from urllib.parse import unquote
import joblib
import pandas as pd
import pytest
import xgboost as xgb


from main import app
from ml.model import inference, train_model, compute_model_metrics, cat_features
from ml.model import transform_data, transform_prediction_attributes, compute_model_metrics
from ml.model_test_helper import load_metrics_from_file
from ml.data import read_and_clean_data

# fixture to create fastAPI client
@pytest.fixture
def client():
   return TestClient(app)


@pytest.fixture
def encoder():
    print("\n".join(sys.path))
    # load encoder from file
    encoder = joblib.load("starter/model/encoder.joblib")
    return encoder


# create fixture to load model
@pytest.fixture
def model():
    print("\n".join(sys.path))
    # load model from file
    model = xgb.Booster()
    model.load_model("starter/model/xgb.model")
    return model


def test_predictions(model):
    """
    Tests that the predictions are the same as the saved predictions.
    """
    # load sample input and predictions from file
    sample_df = pd.read_csv("starter/data/sample_input_and_predictions.csv")
    
    # get sample input and predictions
    X_test = sample_df.drop("predictions", axis=1)
 
    # make predictions
    predictions = inference(model, X_test)
    
    # load saved predictions
    saved_predictions = sample_df["predictions"]
        
    # test that predictions are the same as saved predictions
    assert predictions == saved_predictions.tolist()

def test_encoder(encoder):
    """
    Tests that the encoder is the same as the saved encoder.
    """
    # load sample input and predictions from file
    sample_df = pd.read_csv("starter/data/sample_raw_input_and_predictions.csv")
    
    # get sample input and predictions
    X_test = sample_df.drop("predictions", axis=1)

    # make predictions
    X_test_encoded = transform_prediction_attributes(X_test, encoder, cat_features)

    # test that the encoded created the right number of columns
    assert X_test_encoded.shape[1] == 65
    
def confirm_model_api_results(dataframe, client):
    # get sample input and predictions
    X_test = dataframe.drop("predictions", axis=1)
    saved_predictions = dataframe["predictions"]
    
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


def test_model_api_post_negative(client):
    """
    Tests that the model API returns the correct predictions.
    """
    # load sample input and predictions from file
    sample_df = pd.read_csv("starter/data/sample_raw_input_and_predictions_negative.csv")
    confirm_model_api_results(sample_df, client)
    
def test_model_api_post_positive(client):
    """
    Tests that the model API returns the correct predictions.
    """
    # load sample input and predictions from file
    sample_df = pd.read_csv("starter/data/sample_raw_input_and_predictions_positive.csv")
    confirm_model_api_results(sample_df, client)


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


def test_model_metrics_computation():
    # Create sample predictions and labels to test compute_model_metrics function
    sample_predictions = [0, 1, 0, 1, 0, 1, 1, 1, 0, 1]
    sample_labels = [0, 0, 0, 1, 1, 1, 0, 1, 0, 0]
    expected_precision = 0.75
    expected_recall = 0.5
    expected_fbeta = 0.6

    # Compute model metrics
    precision, recall, fbeta = compute_model_metrics(sample_predictions, sample_labels)  
    
    assert expected_precision == precision
    assert expected_recall == recall
    assert expected_fbeta == fbeta


