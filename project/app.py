# Load the libraries
import uvicorn
import pandas as pd
import xgboost as xgb

from fastapi import FastAPI
from pydantic import BaseModel
from preprocessing.preprocessing import process_telecom_data

class CustomerDetails(BaseModel):
    """
    pydantic class to validate API inputs
    """
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str                           # can still pass non string as string
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class DefaultPrediction(BaseModel):
    """
    pydantic class to validate API output
    """
    Default: float


# Load the model
model = xgb.XGBClassifier()
model.load_model('model/xgboost_model.pickle.dat')


# Initialize an instance of FastAPI
app = FastAPI()


# Define the default route
@app.get("/")
def root():
    """
    Homepage:

    display welcome message
    """
    return {"message": "Welcome to Your Finantier Assessment FastAPI"}


@app.post("/predict", response_model=DefaultPrediction)
def predict_default(inputs: CustomerDetails):
    """
    predict endpoint:

    return predicted Default probability given customer data
    following inputs required by the CustomerDetail validator
    """
    # convert data into dataframe
    inputs = inputs.dict()
    df = pd.DataFrame([inputs])

    # preprocess and add features
    print(df)
    df = process_telecom_data(df, 'preprocessing/encoder.pickle')
    print(df.columns)

    # predict sales
    pred = model.predict_proba(df)[::,1]

    return {
        "Sale": pred
    }


# Only use below 2 lines when testing on localhost -- remove when deploying
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
