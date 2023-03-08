import json
from numpy import random
import pandas as pd

def save_metrics_to_file(precision, recall, fbeta, accuracy):
    """
    Saves the model metrics to a file.
    """
    
    # create a dictionary of metrics
    metrics = { 
        "precision": precision,
        "recall": recall,
        "fbeta": fbeta,
        "accuracy": accuracy
    }
    
    # save metrics to file as json  
    with open("starter/tests/metrics.json", "w") as f:
        json.dump(metrics, f)
        
        
def load_metrics_from_file():
    """
    Loads the model metrics from a file.
    """
    with open("starter/tests/metrics.json", "r") as f:
        metrics = json.load(f)
    return metrics


def save_sample_input_and_predictions_to_file(X_test, predictions):
    """
    Saves the sample input and predictions to a file.
    
    Inputs
    ------
    X_test : np.array
        Test data to sample from.
    predictions : np.array
        predictions to sample from.
    """
    
    # add predictions to x_test
    df = pd.DataFrame(X_test)
    df["predictions"] = predictions
    
    # sample data from X_test and predictions
    sample_df = df.sample(n=10, random_state=31)
    
    # save sample input and predictions to file as csv
    sample_df.to_csv("starter/tests/sample_input_and_predictions.csv", index=False)
    
        

    