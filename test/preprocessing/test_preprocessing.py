import pytest
import pandas as pd
from tables import Cols
from yaml import load
from project.preprocessing.preprocessing import load_data, process_telecom_data




class TestLoadData(object):
    def test_on_complete_columns(self):
        df = load_data('project/data/finantier_data_technical_test_dataset.csv')
        
        cols = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Default']

        is_complete = df.columns.to_list() == cols

        assert is_complete, "data is loaded incompletely, should contain the ff: {}".format(cols)
    
    def test_on_no_erroneous_rows(self):
        df = load_data('project/data/finantier_data_technical_test_dataset.csv')
        
        is_no_emptyrow = ~df.isnull().all(axis=1).any()

        assert is_no_emptyrow, "data contains empty rows"

class TestProcessTelecomData(object):
    def test_on_complete_columns(self):
        df = load_data('project/data/finantier_data_technical_test_dataset.csv')
        df = process_telecom_data(df)

        cols = ['gender_Male', 'SeniorCitizen_1.0', 'Partner_Yes', 'Dependents_Yes',
        'PhoneService_Yes', 'MultipleLines_No phone service',
        'MultipleLines_Yes', 'InternetService_Fiber optic',
        'InternetService_No', 'OnlineSecurity_No internet service',
        'OnlineSecurity_Yes', 'OnlineBackup_No internet service',
        'OnlineBackup_Yes', 'DeviceProtection_No internet service',
        'DeviceProtection_Yes', 'TechSupport_No internet service',
        'TechSupport_Yes', 'StreamingTV_No internet service', 'StreamingTV_Yes',
        'StreamingMovies_No internet service', 'StreamingMovies_Yes',
        'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
        'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
        'customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Default']
        
        is_complete = df.columns.to_list() == Cols
        
        assert is_complete, "data is processed incompletely, should contain the ff: {}".format(cols)
