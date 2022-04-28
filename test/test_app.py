import pytest
from project.app import root, predict_default


sample_input = {
    "customerID": "7590-VHVEG",
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",                           # can still pass non string as string
    "Dependents": "No",
    "tenure": 32,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "One year",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 64.762,
    "TotalCharges": 2279.7155,
    }

class TestRoot(object):
    def test_return_welcome_message(self):
        test = root()

        is_responding = isinstance(test, dict)

        assert is_responding, "homepage is not returning message"

class TestProcessTelecomData(object):
    def test_on_returning_prediction(self):
        test = predict_default(sample_input)
        
        is_prediction = test['Default_prob']
        
        assert is_prediction is not None , "no predictions"
