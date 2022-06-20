#Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import scipy.stats as stats
import statistics

import matplotlib.pyplot as plt 
import seaborn as sns

import wrangle as wr
import explore as ex

np.random.seed(4)

from sklearn.cluster import KMeans


def df_distribution(df):
    """Takes a dataframe and displays the distribution of each feature"""
    columns = df.columns
    for col in columns:
        plt.figure(figsize=(10,8))
        sns.displot(data=df.sample(1000), x=col)
    return


def split_data():
    """Splits data into train, validate, test. Cleans columns. Only observes specified columns. Remaps needed columns."""
    
        # split the data
    train, validate, test = wr.wrangle_zillow()

    cols = ['parcelid','logerror', 'bathroomcnt',
                         'bedroomcnt', 'calculatedfinishedsquarefeet',
                         'fips', 'yearbuilt', 'propertylandusedesc', 'taxvaluedollarcnt']

    train = train[cols]
    validate = validate[cols]
    test = test[cols]   

    train = train.rename(columns={'bathroomcnt':'bathrooms', 
                                              'bedroomcnt':'bedrooms',
                                              'calculatedfinishedsquarefeet':'square_feet',
                                              'fips':'county', 'yearbuilt':'year_built', 
                                              'propertylandusedesc':'property_type', 
                                              'taxvaluedollarcnt': 'home_value'})

    validate = validate.rename(columns={'bathroomcnt':'bathrooms', 
                                              'bedroomcnt':'bedrooms',
                                              'calculatedfinishedsquarefeet':'square_feet',
                                              'fips':'county', 'yearbuilt':'year_built', 
                                              'propertylandusedesc':'property_type', 
                                        'taxvaluedollarcnt': 'home_value'})

    test = test.rename(columns={'bathroomcnt':'bathrooms', 
                                              'bedroomcnt':'bedrooms',
                                              'calculatedfinishedsquarefeet':'square_feet',
                                              'fips':'county', 'yearbuilt':'year_built', 
                                              'propertylandusedesc':'property_type', 
                                'taxvaluedollarcnt': 'home_value'})

    # set counties
    county = {6037:'Los Angeles', 6059:'Orange', 6111:'Ventura'}

    train.county = train.county.map(county)

    validate.county = validate.county.map(county)

    test.county = test.county.map(county)




    # changing data types
    train.bedrooms = train.bedrooms.astype(int)
    train.square_feet = train.square_feet.astype(int)
    train.year_built = train.year_built.astype(int)

    validate.bedrooms = validate.bedrooms.astype(int)
    validate.square_feet = validate.square_feet.astype(int)
    validate.year_built = validate.year_built.astype(int)

    test.bedrooms = test.bedrooms.astype(int)
    test.square_feet = test.square_feet.astype(int)
    test.year_built = test.year_built.astype(int)


    # omit outliers
    cols = ['bathrooms', 'bedrooms', 'square_feet']

    train = wr.omit_outliers(train, 1.5, cols)
    validate = wr.omit_outliers(validate, 1.5, cols)
    test = wr.omit_outliers(test, 1.5, cols)
    return train, validate, test


def train_data_distribution(df):
    """Takes in a train dataframe and returns the distribution of specified columns"""
    
    columns = ['bathrooms','bedrooms', 'square_feet', 'year_built',
           'property_type', 'county', 'home_value']

    for col in columns:
        plt.figure(figsize=(10,8))
        sns.displot(data=df.sample(1000), x=col)
    return


def corr_graph(df):
    plt.figure(figsize = (12,8))
    df.corr()['logerror'].sort_values(ascending=False).plot(kind='barh', color='orange')
    plt.title('Relationship with Logerror')
    plt.xlabel('Relationship')
    plt.ylabel('Features')
    return


def set_home_age(train, validate, test):
    """Calculates and set the age of each home"""
    train['property_age'] = (2017 - train.year_built)
    validate['property_age'] = (2017 - validate.year_built)
    test['property_age'] = (2017 - test.year_built)
    return 


def bath_and_logerror(df):
    """Creates a lmplot for bathroom and logerror for the zillow dataset"""
    sns.lmplot(x='bathrooms', y='logerror', data=df, line_kws={'color':'red'})
    plt.show()
    return

def beds_and_log(df):
    """Creates an lmplot for bedrooms and logerror"""
    sns.lmplot(x='bedrooms', y='logerror', data=df, line_kws={'color':'red'})
    plt.show()
    return

def age_and_logerror(df):
    plt.figure(figsize=(16,8))
    sns.lmplot(data=df, x='property_age', y='logerror', line_kws={'color':'red'})
    plt.show
    plt.title('Correlation between Property Age and Logerror')
    return


    

    
    



        


