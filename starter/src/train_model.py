# Script to train machine learning model.

import logging
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
import numpy as np
import sys

print("\n".join(sys.path))  

from ml.data import process_data
from ml.data import read_and_clean_data
from ml.model import train_model
from ml.model import compute_model_metrics
from ml.model import inference
from ml.model import transform_data
from ml.model import cat_features
from ml.model_test_helper import save_metrics_to_file
from ml.model_test_helper import save_sample_input_and_predictions_to_file

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)   # set the logging level
# create file handler to log messages to file
file_handler = logging.FileHandler('starter/logs/train_model.log')
logger.addHandler(file_handler)


# Read and transform the data
df = read_and_clean_data('starter/data/census.csv')
X_train, y_train, X_test, y_test, X_test_raw, encoder = transform_data(df)

# train xgboost model, save it and calculate accuracy metrics
model = train_model(X_train, y_train)
predictions = inference(model, X_test)

def calculate_save_model_metrics(logger, X_test, y_test, X_test_raw, predictions):
    precision, recall, fbeta = compute_model_metrics(y_test, predictions)  

    # calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    logger.info("Accuracy: %.2f%%" % (accuracy * 100.0))

    #log metrics
    logger.info(f'Precision: {precision}')
    logger.info(f'Recall: {recall}')
    logger.info(f'F1: {fbeta}')

    # calculate confusion matrix and classification report
    logger.info(f'Confusion matrix: {confusion_matrix(y_test, predictions)}')
    logger.info(f'Classification Report: {classification_report(y_test, predictions)}')

    # save metrics to file
    save_metrics_to_file(precision, recall, fbeta, accuracy, 
                         "starter/tests/metrics.json", "w", None, None)

    #save sample input and predictions to file
    save_sample_input_and_predictions_to_file(X_test_raw, X_test, predictions)
    
    
def calculate_save_model_metrics_by_slice(logger, X_test_raw, y_test):
    # concatenate predictions with test data
   
    test_df = pd.concat([pd.DataFrame(X_test_raw), pd.DataFrame(y_test, columns=['income'])
                         ], axis=1)
    print(test_df.columns)
    test_df.dropna(inplace=True)
    # Check if the DataFrame has missing values
    if test_df.isna().any().any():
        print("The DataFrame has missing values.")
    else:
        print("The DataFrame does not have missing values.")
    for cat_feature in cat_features:
        # get all unique values for a category
        categories = test_df[cat_feature].unique()
        for cat in categories:
            # get all rows for a category
            cat_df = test_df[test_df[cat_feature] == cat]
            # get predictions and labels for a category
            cat_predictions = cat_df['predictions']
            cat_labels = cat_df['income']
            # calculate metrics for a category
            precision, recall, fbeta = compute_model_metrics(cat_labels, cat_predictions)  
            accuracy = accuracy_score(cat_labels, cat_predictions)
            logger.info(f'Accuracy for {cat_feature} {cat}: {accuracy}')
            logger.info(f'Precision for {cat_feature} {cat}: {precision}')
            logger.info(f'Recall for {cat_feature} {cat}: {recall}')
            logger.info(f'F1 for {cat_feature} {cat}: {fbeta}')
            #  save metrics to file
            save_metrics_to_file(precision, recall, fbeta, accuracy, 
                                 "starter/tests/slice_output.txt", "a", cat_feature, cat)
  
calculate_save_model_metrics(logger, X_test, y_test, X_test_raw, predictions)
calculate_save_model_metrics_by_slice(logger, X_test_raw, y_test)

# save model
model.save_model("starter/model/xgb.model")
# save encoder
joblib.dump(encoder, 'starter/model/encoder.joblib')

# load model
model = xgb.Booster()
model.load_model("starter/model/xgb.model")







