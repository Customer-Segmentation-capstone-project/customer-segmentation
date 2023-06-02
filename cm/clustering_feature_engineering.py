import wrangle as w
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
# visualizations
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit

df, train, validate, test = w.wrangle_data()

def KMeans_model_dict(train):
    '''
    This functions creates numerous KMeans models with various k values, fits it on the specified features from the train dataset, 
    and then ceerates a dictionary of inertias for each model. 
    '''

    # Specifying the features to train the KMeans model

    X = train[['customer_age', 'quantity', 'unit_cost',
       'unit_price', 'cost', 'revenue', 'total_price',
       'sub_category_Bike Stands', 'sub_category_Bottles and Cages',
       'sub_category_Caps', 'sub_category_Cleaners', 'sub_category_Fenders',
       'sub_category_Gloves', 'sub_category_Helmets',
       'sub_category_Hydration Packs', 'sub_category_Jerseys',
       'sub_category_Mountain Bikes', 'sub_category_Road Bikes',
       'sub_category_Shorts', 'sub_category_Socks',
       'sub_category_Tires and Tubes', 'sub_category_Touring Bikes',
       'sub_category_Vests', 'customer_gender_M', 'country_Germany',
       'country_United Kingdom', 'country_United States']]
    
    # Creating models with various k values and storing the inertia values in a dictionary
    my_kmeans_dict = {}
    for k in range(1,15):
        my_kmeans_dict[k] = KMeans(k).fit(X).inertia_
    
    return my_kmeans_dict

def visualize_KMeans_inertia():

    '''
    This function uses the dictionary of inertia values from the KMeans_model_dict function to visualize the change in the inertia
    as the number of k (the specified number of clusters) increases. This illustrates the elbow method which helps find the optimal 
    k value to use in clustering modeling.
    '''

    my_kmeans_dict = KMeans_model_dict(train)

    plt.figure(figsize=(12,12))
    pd.Series(my_kmeans_dict).plot(marker='x')
    plt.title('Elbow Method Visualization')
    plt.xlabel('k value (number of clusters)')
    plt.ylabel('inertia')
    plt.show()

def engineer_clusters(train, validate, test, k):

    '''
    This function uses the specified k value (number of clusters) to create a model, fit the model on specified modeling features, 
    and then predict clusters of each transaction (row). The function then adds the clsuters back to the split datasets as newly
    engineered features.
    '''

    # specifying features used in modeling
    modeling_feats = ['customer_age', 'quantity', 'unit_cost',
       'unit_price', 'cost', 'revenue', 'total_price',
       'sub_category_Bike Stands', 'sub_category_Bottles and Cages',
       'sub_category_Caps', 'sub_category_Cleaners', 'sub_category_Fenders',
       'sub_category_Gloves', 'sub_category_Helmets',
       'sub_category_Hydration Packs', 'sub_category_Jerseys',
       'sub_category_Mountain Bikes', 'sub_category_Road Bikes',
       'sub_category_Shorts', 'sub_category_Socks',
       'sub_category_Tires and Tubes', 'sub_category_Touring Bikes',
       'sub_category_Vests', 'customer_gender_M', 'country_Germany',
       'country_United Kingdom', 'country_United States']

    # creating the moddel object
    k_means_model = KMeans(n_clusters=k)
    # fitting the model to the train features
    k_means_model.fit(train[modeling_feats])
    # predicting clusters on train, validate, and test
    train_clusters = k_means_model.predict(train[modeling_feats])
    validate_clusters = k_means_model.predict(validate[modeling_feats])
    test_clusters = k_means_model.predict(test[modeling_feats])

    # Adding the clusters back to the split datasets as newly engineered features
    train['clusters'] = train_clusters
    validate['clusters'] = validate_clusters
    test['clusters'] = test_clusters

    return train, validate, test
