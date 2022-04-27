# Load the libraries
import uvicorn
import pandas as pd
import xgboost as xgb

from fastapi import FastAPI
from pydantic import BaseModel
from preprocessing.preprocessing import merge_features, process


class StoreDetails(BaseModel):
    """
    pydantic class to validate API inputs
    """
    Store: int
    DayOfWeek: int
    Date: str                           # can still pass non string as string
    Customers: int
    Open: int
    Promo: int
    StateHoliday: str
    SchoolHoliday: int


class SalesPrediction(BaseModel):
    """
    pydantic class to validate API output
    """
    Sale: float


# Load the model
model = xgb.XGBRegressor()
model.load_model('model/xgboost_model.txt')

# Load the store data
df_store = pd.read_pickle('data/store_features.pickle')


# Initialize an instance of FastAPI
app = FastAPI()


# Define the default route
@app.get("/")
def root():
    """
    Homepage:

    display welcome message
    """
    return {"message": "Welcome to Your GCash Assessment FastAPI"}


@app.post("/predict", response_model=SalesPrediction)
def predict_sales(inputs: StoreDetails):
    """
    predict endpoint:

    return predicted sales given store data
    following inputs required by the StoreDetails validator
    """
    # convert data into dataframe
    inputs = inputs.dict()
    df = pd.DataFrame([inputs])

    # preprocess and add features
    print(df)
    df = merge_features(df, df_store)
    df = process(df, isTest=True)
    print(df.columns)

    # predict sales
    pred = model.predict(df)

    # add heuristics to results
    pred = pred.clip(min=0)
    if df.Open[0]==0:
        pred = 0

    return {
        "Sale": pred
    }


# Only use below 2 lines when testing on localhost -- remove when deploying
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
