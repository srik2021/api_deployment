import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder



def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)
    
    #impute missing continuous variables as median
    

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb

def transform_prediction_attributes(X, encoder, categorical_features):
    """
    Transform the prediction attributes.

    Args:
        X (pd.Dataframe): data of input attributes to be transformed
        encoder (_type_): encoding to perform on th categorical features.  
        Should be the same as was used in training.
        categorical_features (list): list of categorical features to be encoded.

    Returns:
        NDArray : concatenated array of continuous and encoded categorical features.
    """    
    
    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)
    
    X_categorical = encoder.transform(X_categorical)
    return np.concatenate([X_continuous, X_categorical], axis=1)


def read_and_clean_data(path):
    """ 
    Read and process the data.

    Inputs
    ------
    path : str
        Path to the data.

    Returns
    -------
    data : pd.DataFrame
        Processed data.
    """
    
    df = pd.read_csv(path)
    df = df.replace("?", np.nan)

    # Remove duplicates from the DataFrame
    df = df.drop_duplicates()

    # Filter data to only include 'United-States' in the native-country column
    df = df[df['native-country'] == 'United-States']

    # Filter age to be between 19 and 69
    df = df[(df['age'] >= 19) & (df['age'] <= 69)]

    return df
