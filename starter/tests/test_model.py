import pandas as pd
import pytest
from sklearn.metrics import accuracy_score
import xgboost as xgb
import sys
      
from src.ml.model import inference, train_model, compute_model_metrics
from src.ml.model_test_helper import load_metrics_from_file
from src.train_model import transform_data
from src.ml.data import read_and_clean_data


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
    sample_df = pd.read_csv("starter/tests/sample_input_and_predictions.csv")
    
    # get sample input and predictions
    X_test = sample_df.drop("predictions", axis=1)
 
    # make predictions
    predictions = inference(model, X_test)
    
    # load saved predictions
    saved_predictions = sample_df["predictions"]
        
    # test that predictions are the same as saved predictions
    assert predictions == saved_predictions.tolist()
    
def test_model_metrics(model):
    """
    Tests that the model metrics are the same as the saved metrics.
    """
    # load input data from file and transform it
    df = read_and_clean_data("starter/data/census.csv")   
    X_train, y_train, X_test, y_test = transform_data(df)
    
   # train xgboost model, save it and calculate accuracy metrics
    model = train_model(X_train, y_train)
    predictions = inference(model, X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, predictions)  

    # calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    
    # load saved metrics
    saved_metrics = load_metrics_from_file()
    
    # test that metrics are the same as saved metrics
    assert saved_metrics["precision"] == precision
    assert saved_metrics["recall"] == recall
    assert saved_metrics["fbeta"] == fbeta
    assert saved_metrics["accuracy"] == accuracy