import json
from numpy import random
import pandas as pd

def save_metrics_to_file(precision, recall, fbeta, accuracy, file_loc, 
                         file_open_mode, feature, feature_value):
    """
    Saves the model metrics to a file.
    """
        
    metrics = { 
        "precision": precision,
        "recall": recall,
        "fbeta": fbeta,
        "accuracy": accuracy
    }

    if feature is not None: 
        metrics["feature"] = feature
        metrics["feature_value"] = feature_value
   
    # save metrics to file as json  
    with open(file_loc, file_open_mode) as f:
        json.dump(metrics, f)
        f.write("\n")
        
        
def load_metrics_from_file():
    """
    Loads the model metrics from a file.
    """
    with open("starter/data/metrics.json", "r") as f:
        metrics = json.load(f)
    return metrics


def save_sample_input_and_predictions_to_file(X_test_raw, X_test, predictions):
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
    
    df_raw = pd.DataFrame(X_test_raw)
    df_raw["predictions"] = predictions
    
    # sample data from X_test and predictions
    sample_df = df.sample(n=10, random_state=31)
    sampl_df_raw = df_raw.sample(n=10, random_state=31)
    
    # save sample input and predictions to file as csv
    sample_df.to_csv("starter/data/sample_input_and_predictions.csv", index=False)
    sampl_df_raw.to_csv("starter/data/sample_raw_input_and_predictions.csv", index=False)
    
        

    