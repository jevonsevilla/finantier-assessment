# finantier-assessment
Continuous deployment of an API to predict the likelihood of customer defaulting on telco payment based on their telco data. In partial fulfillment of my Finantier application.

API LINK: https://finantier-assessment.herokuapp.com/predict

# Demo-Preview
![image](https://user-images.githubusercontent.com/52987305/165675854-1758e6c7-9980-42e9-89b2-04b44ac4aec5.png)

requests to the API can be completed as shown.

## Example Request
```
        import requests


        r = requests.post('https://finantier-assessment.herokuapp.com/predict', json = {
              "customerID": "7590-VHVEG",                     # fields take sting unless noted otherwise
              "gender": "Male",                              
              "SeniorCitizen": 0,                             # takes int (0,1)
              "Partner": "Yes",                              
              "Dependents": "No",                             
              "tenure": 32,                                   # takes int
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
              "MonthlyCharges": 64.762,                       # takes float
              "TotalCharges": 2279.7155                       # takes float
              })


        print(r.text)
```
## Example Output
```
        {"Default_prob":0.3874466121196747}
```

# Table of contents
- [Project Title](#project-title)
- [Demo-Preview](#demo-preview)
    - [Example Request](#example-request)
    - [Example Output](#example-output)
- [Table of contents](#table-of-contents)
- [Usage](#usage)
- [Development](#development)

# Usage
## Current Model Performance
> Results on hold out test set  
> **AUC: 0.859**
<!-- This is optional and it is used to give the user info on how to use the project after installation. This could be added in the Installation section also. -->

## Local Machine
After downloading and installing [docker daemon](https://docs.docker.com/get-docker/). The following commands will be able to deploy a docker container in your local machine. 
```
        >>> cd project
        >>> cd pip install -r requirements.txt
        >>> docker build -t fastapiapp:latest -f Dockerfile .
        >>> docker run -p 80:80 fastapiapp:latest
```

# Development and CI/CD
Push and Pull Requests to the masterbranch trigger Github Actions that automatically deploy the application to Heroku. 
The app that is deployed is built in docker and utilizes FastAPI to serve predictions made with an XGBoost Model.

The application and its components are developed in the JupyterNotebook. Outputs are then saved in the relevant directories in order to make changes to the model.

relevant outputs:  
`project/model/xgboost_model.pickle.dat`: saved model that was fit using *gridsearchCV* with **recall** as the scoring method to search for the best hyperparameters  
`project/preprocessing/encoder.pickle`: saved *OneHotEncoder* object to apply the fitted encodings to future datasets.

