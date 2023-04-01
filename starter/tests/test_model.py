import pandas as pd
import pytest
from sklearn.metrics import accuracy_score
import xgboost as xgb
import sys


print("\n".join(sys.path))
      
from ml.model import inference, train_model, compute_model_metrics, transform_data
from ml.model_test_helper import load_metrics_from_file
from ml.data import read_and_clean_data


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
    
