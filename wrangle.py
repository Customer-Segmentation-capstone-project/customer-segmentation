import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# =================================================================================

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

def encode_cat_variables(df, encode_cols):
    '''
    This will encode the passed categorical columns into numerical values
    '''
    # encode categorical columns into numerical values that can be used in modeling
    # create encoder object
    le = LabelEncoder()
    for col in encode_cols:
        le.fit(df[col])
        # create a new column with the encoded values
        df[f'{col}_encoded'] = le.transform(df[col])
    # return the df with the encoded column
    return df

# =================================================================================

def one_hot_encode_columns(df, cols):
    '''
    This will one hot encode the passed categorical columns
    '''
    # one-hot encode the passed column
    df = pd.concat([df, 
                    pd.get_dummies(df[[cols]], 
                     dummy_na=False, 
                     drop_first=[True, True])], axis=1)
    # return the df with the one-hot encoded columns
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
    
    # set date to datetime dtype
    df.date = pd.to_datetime(df.date)
    
    # create new column for total price of sale
    df['total_price'] = df.quantity * df.unit_price
    
    # one-hot encode 'sub-category' column
    df = one_hot_encode_columns(df, 'sub_category')
    # one-hot encode 'customer_gender' column
    df = one_hot_encode_columns(df, 'customer_gender')
    # one-hot encode 'country' column
    df = one_hot_encode_columns(df, 'country')
    
    #encode categorical varibles
    encode_cols = ['product_category']
    df = encode_cat_variables(df, encode_cols)
    
    # return the cleaned dataset
    return df

# =================================================================================

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

# =================================================================================

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

# =================================================================================

def scale_data(train,
               validate,
               test,
               columns_to_scale=['customer_age'],
               scaler=StandardScaler(),
               return_scaler=False):
    '''
    Scales the 3 data splits. 
    Takes in train, validate, and test data splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''
    # make copies of our original data so we dont gronk up anything
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    #     fit the thing
    scaler.fit(train[columns_to_scale])
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(
        scaler.transform(train[columns_to_scale]),
        columns=train[columns_to_scale].columns.values, 
        index = train.index)
    # apply the scaller on the validation dataset                                    
    validate_scaled[columns_to_scale] = pd.DataFrame(
        scaler.transform(validate[columns_to_scale]),
        columns=validate[columns_to_scale].columns.values).set_index(
        [validate.index.values])
    # apply the scaler on the test dataset
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(
        test[columns_to_scale]), 
        columns=test[columns_to_scale].columns.values).set_index(
        [test.index.values])
    
    # if we requested the scaler returned in the function call,
    # then return the scaler object along with the scaled data
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    # otherwise return the scaled data
    else:
        return train_scaled, validate_scaled, test_scaled

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

def k_means_clustering(df):
    # Drop date column from dataset
    df.drop('date', axis=1, inplace=True)

    # Encode categorical columns
    le = LabelEncoder()
    df['customer_gender'] = le.fit_transform(df['customer_gender'])
    df['country'] = le.fit_transform(df['country'])
    df['state'] = le.fit_transform(df['state'])
    df['product_category'] = le.fit_transform(df['product_category'])
    df['sub_category'] = le.fit_transform(df['sub_category'])
    df['month'] = le.fit_transform(df['month'])

    # Standardizes data
    df = (df - df.mean()) / df.std()

    # Feature Selection
    data = df[['customer_age','sub_category', 'cost']]  # Select relevant columns

    # Standardization
    scaler = StandardScaler()
    X = scaler.fit_transform(data)

    # Clustering Algorithm (K-means)
    k = 5  # Number of clusters
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(X)

    # Cluster Analysis
    # Analyze the resulting clusters by examining their characteristics
    cluster_centers = kmeans.cluster_centers_

    # Print the mean values of the selected features for each cluster
    for cluster in range(k):
        cluster_data = data[labels == cluster]
        cluster_mean = cluster_data.mean()
        print(f"Cluster {cluster + 1} Mean:")
        print(cluster_mean)
        print()
    # Print silhouette_score for clusters     
    s_score = silhouette_score(X, labels)
    print(f"Silhouette Score: {s_score:.3f}")
    
# Execute with the following statement    
# k_means_clustering(df)