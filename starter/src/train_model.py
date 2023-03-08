# Script to train machine learning model.

import logging
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd

from src.ml.data import process_data
from src.ml.data import read_and_clean_data
from src.ml.model import train_model
from src.ml.model import compute_model_metrics
from src.ml.model import inference
from src.ml.model import transform_data
from src.ml.model_test_helper import save_metrics_to_file
from src.ml.model_test_helper import save_sample_input_and_predictions_to_file

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)   # set the logging level
# create file handler to log messages to file
file_handler = logging.FileHandler('starter/logs/train_model.log')
logger.addHandler(file_handler)


# Read and transform the data
df = read_and_clean_data('starter/data/census.csv')
X_train, y_train, X_test, y_test = transform_data(df)

# train xgboost model, save it and calculate accuracy metrics
model = train_model(X_train, y_train)
predictions = inference(model, X_test)

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

# save model
model.save_model("starter/model/xgb.model")

# load model
model = xgb.Booster()
model.load_model("starter/model/xgb.model")

# save metrics to file
save_metrics_to_file(precision, recall, fbeta, accuracy)

#save sample input and predictions to file
save_sample_input_and_predictions_to_file(X_test, predictions)





