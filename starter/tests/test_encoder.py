# create fixture to load model
import sys
import joblib
import pandas as pd
import pytest
from ml.data import transform_prediction_attributes
from ml.model import transform_data
from ml.model import cat_features


@pytest.fixture
def encoder():
    print("\n".join(sys.path))
    # load encoder from file
    encoder = joblib.load("starter/model/encoder.joblib")
    return encoder

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
    