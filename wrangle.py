from env import host, user, password, get_db_url
import pandas as pd 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

def acquire(use_cache=True):
    filename = 'zillow.csv'

    if os.path.isfile(filename) and use_cache:
        return pd.read_csv(filename)
    else:
        df = pd.read_sql('''
    SELECT
        prop.*,
        predictions_2017.logerror,
        predictions_2017.transactiondate,
        air.airconditioningdesc,
        arch.architecturalstyledesc,
        build.buildingclassdesc,
        heat.heatingorsystemdesc,
        landuse.propertylandusedesc,
        story.storydesc,
        construct.typeconstructiondesc
    FROM properties_2017 prop
    JOIN (
        SELECT parcelid, MAX(transactiondate) AS max_transactiondate
        FROM predictions_2017
        GROUP BY parcelid) pred USING(parcelid)
    JOIN predictions_2017 ON pred.parcelid = predictions_2017.parcelid AND pred.max_transactiondate = predictions_2017.transactiondate
    LEFT JOIN airconditioningtype air USING (airconditioningtypeid)
    LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid)
    LEFT JOIN buildingclasstype build USING (buildingclasstypeid)
    LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid)
    LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid)
    LEFT JOIN storytype story USING (storytypeid)
    LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid)
    WHERE prop.latitude IS NOT NULL AND prop.longitude IS NOT NULL AND transactiondate <= '2017-12-31';''' , get_db_url('zillow'))
        df.to_csv(filename, index=False)
    return df

def nulls_data(df):
    nulls = df.isnull().sum()
    rows = len(df)
    percent_missing = nulls / rows 
    dataframe = pd.DataFrame({'rows_missing': nulls, 'percent_missing': percent_missing})
    return dataframe

def null_cols(df):
    new_df = pd.DataFrame(df.isnull().sum(axis=1), columns = ['cols_missing']).reset_index()\
    .groupby('cols_missing').count().reset_index().\
    rename(columns = {'index': 'rows'})
    new_df['percent_missing'] = new_df.cols_missing/df.shape[1]
    return new_df

def get_single_unit_homes(df):
    single_unit = [261, 262, 263, 264, 266, 268, 273, 276, 279]
    df = df[df.propertylandusetypeid.isin(single_unit)]
    return df
    
def handle_missing_values(df, prop_required_column = .67, prop_required_row = .75):
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df


def split_data(df):
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    return train, validate, test

## dictionary to be used in imputing_missing_values function
impute_strategy = {
'mean' : [
       'calculatedfinishedsquarefeet',
       'finishedsquarefeet12',
     'structuretaxvaluedollarcnt',
        'taxvaluedollarcnt',
        'landtaxvaluedollarcnt',
        'taxamount'
    ],
    'most_frequent' : [
        'calculatedbathnbr',
         'fullbathcnt',
        'regionidcity',
         'regionidzip',
         'yearbuilt'
     ],
     'median' : [
         'censustractandblock'
     ]
 }

def impute_missing_values(df, impute_strategy):
    train, validate, test = split_data(df)
    
    for strategy, columns in impute_strategy.items():
        imputer = SimpleImputer(strategy = strategy)
        imputer.fit(train[columns])

        train[columns] = imputer.transform(train[columns])
        validate[columns] = imputer.transform(validate[columns])
        test[columns] = imputer.transform(test[columns])
    
    return train, validate, test



def prepare_zillow(df):
    '''Prepare zillow for data exploration.'''
    df = get_single_unit_homes(df)
    df = handle_missing_values(df)
    train, validate, test = impute_missing_values(df, impute_strategy)
    return train, validate, test
    
def wrangle_zillow():
    '''Acquire and prepare data from Zillow database for explore'''
    train, validate, test = prepare_zillow(acquire())
    
    return train, validate, test


def omit_outliers(df, stdev, columns):
    for col in columns:
        
        # select quartiles
        q1, q3 = df[col].quantile([.25,.75]) 
        
        # calculate interquartile range
        iqr = q3 - q1
        
        upper_bound = q3 + stdev * iqr
        lower_bound = q1 - stdev * iqr
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
    return df


def import_observed_columns(df):
    df = df[['parcelid','logerror', 'bathroomcnt',
                     'bedroomcnt', 'calculatedfinishedsquarefeet',
                     'fips', 'yearbuilt', 'propertylandusedesc']]
    return df


def scale_data(train, validate, test, return_scaler=False):
    """
    Scales split data
    
    If scaler = True, scaler object will be returned. Set to False. 
    """
    
    # scale the data
    scaled_cols = ['bedrooms', 'bathrooms', 'square_feet', 'property_age', 'year_built']

    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()

    scaler = MinMaxScaler()
    scaler.fit(train[scaled_cols])
    
    # now to transform

    train_scaled[scaled_cols] = scaler.transform(train[scaled_cols])
    validate_scaled[scaled_cols] = scaler.transform(validate[scaled_cols])
    test_scaled[scaled_cols] = scaler.transform(test[scaled_cols])
    
    if return_scaler:
        return train_scaled, validate_scaled, test_scaled, scaler
    else:
        return train_scaled, validate_scaled, test_scaled


