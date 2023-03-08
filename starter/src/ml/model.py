from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
import pandas as pd

from src.ml.data import process_data
from src.ml.data import transform_prediction_attributes

cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

encoder = OneHotEncoder()   

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains an XGBoost machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    param = {
        "max_depth": 5,
        "eta": 0.1,
        "objective": "binary:logistic",
        "eval_metric": "auc"
    }
    num_round = 100
    model = xgb.train(param, dtrain, num_round)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta

def create_prediction_attributes(df):   
    """
    Creates a dataframe with the attributes that will be used for prediction.
    Args:
        df (str): dataframe with raw attributes that will be used for prediction.
    Returns:
        df (pd.DataFrame): Dataframe with transformed attributes.
    """
    return transform_prediction_attributes(df, encoder=encoder, categorical_features=cat_features)
    

def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : xgboost model
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model as boolean values.
    """

    dtest = xgb.DMatrix(X)
    preds = model.predict(dtest)
    preds = [round(value) for value in preds]
    return preds


def transform_data(data):
    """
    Transforms the data into a format that can be used by the model.
    Args:
        data (str): Path to the file containing the data.

    Returns:
        X_train (np.array): Training data.
        y_train (np.array): Training labels.
        X_test (np.array): Test data.
        y_test (np.array): Test labels.
    """  
        
    # Optional enhancement, use K-fold cross validation instead of a train-test split.  
    train, test = train_test_split(data, test_size=0.20, random_state=31, stratify=data["salary"])

    # Process the training data with the process_data function.
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )       

    # Proces the test data with the process_data function.
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False,
        encoder=encoder, lb=lb
    )
    
    return X_train, y_train, X_test, y_test

