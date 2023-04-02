import requests
import json

input_val = {
        "age": 37,
        "workclass": "Private",
        "fnlgt": 167735,
        "education": "Assoc-voc",
        "education_num": 11,
        "marital_status": "Married-civ-spouse",
        "occupation": "Craft-repair",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 1887,
        "hours_per_week": 40,
        "native_country": "United-States"
    }

#create test to invoke api using requests
def test_salary_predict_api():
    response = requests.post("https://salary-predictor.herokuapp.com/predict_salary", 
                             json=input_val)
    assert(response.status_code == 200)
    assert(response.json().get('prediction') == 1)
 