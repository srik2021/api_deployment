# Model Card
The model has been trained on the publicly available dataset (https://www.kaggle.com/datasets/laleeth/salary-predict-dataset).  The model predicts whether a person earns >$50K based on demographic and other information about the person

## Model Details
The model has been trained using XGB.  The model is hosted as an API at https://salary-predictor.herokuapp.com/.   

## Intended Use
The model should only be used as baseline model to understand basic transformations to consider.  Also, the implementation provides some examples of ML engineering best practices like writing modular code and implementing CI/CD

## Training Data
The traning data is from the publicly available dataset: https://www.kaggle.com/datasets/laleeth/salary-predict-dataset

## Evaluation Data
The evaluation data is a subset of the training data - the data was split into training and validation data.  Additional test data was not used to evaluate expected performance

## Metrics
Since this is a classification model,  the standard classification metrics of precision, rcall, fbeta and accuracy were used.  Metric results below:
precision: 0.787719298245614
recall: 0.6355272469922152
fbeta: 0.7034860947904427
accuracy: 0.8632586705202312

## Ethical Considerations


## Caveats and Recommendations
The focus of the code has been on implementing CI/CD best practices rather than producing the best performing model.   Recommend exploring k-fold cross validation to assess if it improves the performance of the model