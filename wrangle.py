import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def acquire_data():
    '''
    Acquire the sales dataset from local cached file or if local file
    does not exits, then download the data from data.world.
    '''
    # set the filename
    filename = 'sales.csv'
    if os.path.exists(filename):
        # if local file exists
        # display status message
        print('Opening data from local file.')
        # open the local data
        df = pd.read_csv(filename)
    else:
        # if the local file does not exist, download the data
        # display status message
        print('Local file not found')
        print('Downloading dataset')
        # download the data from data.world
        df = pd.read_excel('https://query.data.world/s/fhfoecpnngvcahqseb5ai6daubh4gk?dws=00000')
        # create a local cache of the data
        df.to_csv(filename, index=False)
        
    # return the acquired data
    return df

def prepare_data(df):
    '''
    this will clean the column names and change the data types to the proper dtypes.
    '''
    # change column names to lowercase and remove spaces
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    # drop the column1 which has unknown data
    df = df.drop(columns='column1')
    # drop the 1 row containing null values in most of the row
    df = df.dropna()
    # change data types to int
    df.year = df.year.astype(int)
    df.customer_age = df.customer_age.astype(int)
    df.quantity = df.quantity.astype(int)
    # round dollar amounts to 2 digits
    df.unit_cost = round(df.unit_cost, 2)
    df.unit_price = round(df.unit_price, 2)
    df.cost = round(df.cost, 2)
    df.revenue = round(df.revenue, 2)
    # create new column for total price of sale
    df['total_price'] = df.quantity * df.unit_cost
    # return the cleaned dataset
    return df

def split_data(df, random_seed=4233):
    '''
    split_data will take in a DataFrame and split it into train, validate and test sets
    random_seed is also asignable (default = 4233 for no reason).
    It will return the data split up for ML models. 
    The return values are: train, validate, test
    '''
    # split our df into train_val and test:
    train_val, test = train_test_split(df,
                                       train_size=0.8,
                                       random_state=random_seed)
    
    # split our train_val into train and validate:
    train, validate = train_test_split(train_val,
                                       train_size=0.7,
                                       random_state=random_seed)
    # return the split DataFrames
    return train, validate, test

def wrangle_data():
    '''
    This will perform the acquire, preparation and split functions with one command

    returns the clean df, train, validate and test
    '''
    # acquire and clean the data
    df = prepare_data(acquire_data())
    # split the data into train, validate and test
    train, validate, test = split_data(df)
    # return the clean df, train, validate and test
    return df, train, validate, test