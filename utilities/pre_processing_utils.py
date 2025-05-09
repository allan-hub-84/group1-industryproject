from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import pandas as pd

# Common Variables which will be used during different pipelines
onehot_cols = ['State', 'Sex', 'RaceEthnicityCategory', 'SmokerStatus']
ordinal_cols = ['AgeCategory', 'GeneralHealth']
yesno_cols = ['PhysicalActivities', 'HadAngina', 'HadStroke', 'HadCOPD', 'HadKidneyDisease',
             'HadArthritis', 'HadDiabetes', 'ChestScan', 'AlcoholDrinkers']
numeric_cols = ['SleepHours', 'HeightInMeters', 'WeightInKilograms', 'BMI']

# Custom order for AgeCategory
age_order = [['18-24', '25-29', '30-34', '35-39', '40-44',
             '45-49', '50-54', '55-59', '60-64', '65-69',
             '70-74', '75-79', '80+']]

# We will not use this - But keeping this for the reference
# Decided to use the pipeline instead of using this function individually on all categorical columns
def encode_column_train_test(train_df, test_df, column_name, encoding_type, custom_order=None):
    """
    Encode a single column in both train and test sets.
    
    Args:
        train_df (DataFrame): Training DataFrame
        test_df (DataFrame): Test DataFrame
        column_name (str): Column to encode
        encoding_type (str): 'onehot' or 'ordinal'
        custom_order (list): For ordinal encoding, custom order of categories
    
    Returns:
        train_encoded_df, test_encoded_df (DataFrames)
    """
    train_col = train_df[[column_name]].copy()
    test_col = test_df[[column_name]].copy()

    if encoding_type == 'onehot':
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        train_encoded = encoder.fit_transform(train_col)
        test_encoded = encoder.transform(test_col)
        
        col_names = encoder.get_feature_names_out([column_name])
    
    elif encoding_type == 'ordinal':
        if custom_order:
            encoder = OrdinalEncoder(categories=[custom_order])
        else:
            encoder = OrdinalEncoder()
            
        train_encoded = encoder.fit_transform(train_col)
        test_encoded = encoder.transform(test_col)
        
        col_names = [f"{column_name}_ordinal"]
    
    else:
        raise ValueError("encoding_type must be either 'onehot' or 'ordinal'")
    
    train_encoded_df = pd.DataFrame(train_encoded, columns=col_names, index=train_df.index)
    test_encoded_df = pd.DataFrame(test_encoded, columns=col_names, index=test_df.index)

    return train_encoded_df, test_encoded_df

def yes_no_transformer(data):
    """
    Convert the Yes and No to numeric values 1 and 0
    
    Args:
        train_df (Column): column name to convert
    
    Returns:
        Converted numerical column 
    """
    return data.replace({'Yes': 1, 'No': 0})
