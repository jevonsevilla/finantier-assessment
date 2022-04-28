import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder

def load_data(fp):
    """
    Load data from filepath. Removes error rows and empty rows in the process
    """

    df = pd.read_csv(fp)

    # drop error rows
    df = df[~df['ErrorIndicator'].notna()]
    df = df.drop('ErrorIndicator', axis=1)
    df = df[~df.isnull().all(axis=1)]                       # null rows due to formatting

    return df

    
def process_telecom_data(df, fp, train_encoding=False, target=False):
    """
    Process and transform data from raw inputs to model ready features.
    Save encodings in filepath for use on future data.
    """
    # TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df.TotalCharges.replace(" ", "0"))

    # drop customerID
    df = df.drop("customerID", axis=1)

    # encode target
    if target:
        mappings = {'No':0, 'Yes':1}
        df['Default'] = df.Default.map(mappings).fillna(0)

    # encode features
    # enumerate categorical columns
    cat_cols = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"]

    df_cat = df[cat_cols]

    if train_encoding:     
        # Instantiating the Scikit-Learn OHE object
        ohe = OneHotEncoder(handle_unknown='ignore', drop='first', sparse=False)
        
        # Fitting the DataFrame to the Scikit-Learn one-hot encoder
        dummies = ohe.fit_transform(df_cat)

        # save trained encoder for future use
        with open(fp, 'wb') as f:
            pickle.dump(ohe, f)

    else:
        # load pickled encoder
        with open(fp, 'rb') as f:
            ohe = pickle.load(f)
            
        dummies = ohe.transform(df_cat)
        
    # Using the output dummies and transformer categories to produce a cleaner looking dataframe
    dummies_df = pd.DataFrame(dummies)
    dummies_df.columns = ohe.get_feature_names_out()

    numeric_df = df.drop(cat_cols, axis=1).reset_index(drop=True)

    df = pd.concat([dummies_df, numeric_df], axis=1)

    return df