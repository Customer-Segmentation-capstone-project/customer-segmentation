#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_mean_revenue(clustered_df, label):
    """
    Plot the mean revenue for each cluster.

    Parameters:
        clustered_df (pandas.DataFrame): DataFrame with the clustered data.
        label (numpy.array): Cluster labels.

    Returns:
        None
    """
    sns.barplot(x=label, y=clustered_df['revenue'], estimator=np.mean)
    plt.xlabel('Cluster')
    plt.ylabel('Mean Revenue')
    plt.title('Mean Revenue for Each Cluster')
    plt.show()
    
# run line below to execute function    
# plot_mean_revenue(clustered_df, label)


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

def most_buying_power(label, revenue):
    """
    Visualize the cluster with the most buying power based on revenue.

    Parameters:
        label (numpy.array): Cluster labels.
        revenue (pandas.Series): Revenue values.

    Returns:
        None
    """
    sns.barplot(x=label, y=revenue, estimator=np.max)
    plt.xlabel('Cluster')
    plt.ylabel('Revenue')
    plt.title('Cluster with the Most Buying Power')
    plt.show()

# run line below to execute function  
# most_buying_power(label, clustered_df['revenue'])


# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt

def popular_subcategory_purchases(label, train):
    """
    Visualize the popular subcategory purchases for each cluster.

    Parameters:
        label (numpy.array): Cluster labels.
        train (pandas.DataFrame): DataFrame with cluster information.

    Returns:
        None
    """
    sns.countplot(y='sub_category', hue=label, data=train)  # horizontal
    plt.ylabel('Subcategory')
    plt.xlabel('Count')
    plt.title('Cluster Visualization of Subcategory Purchases')
    plt.legend(title='Cluster', loc='lower right')  # leagend lower right for horizontal chart
    plt.show()

    
# popular_subcategory_purchases(label, train)

def top_five_popular_subcategory_purchases(label, train):
    """
    Visualize the popular subcategory purchases for each cluster.

    Parameters:
        label (numpy.array): Cluster labels.
        train (pandas.DataFrame): DataFrame with cluster information.

    Returns:
        None
    """
    # Calculate the counts for each subcategory
    subcategory_counts = train['sub_category'].value_counts()

    # Get the top 5 subcategories
    top_subcategories = subcategory_counts[:5].index

    # Filter the train DataFrame to only include the top 5 subcategories
    train_top_subcategories = train[train['sub_category'].isin(top_subcategories)]

    sns.countplot(y='sub_category', hue=label, data=train_top_subcategories)  # horizontal
    plt.ylabel('Subcategory')
    plt.xlabel('Count')
    plt.title('Cluster Visualization of Top 5 Subcategory Purchases')
    plt.legend(title='Cluster', loc='lower right')  # legend lower right for horizontal chart
    plt.show()

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

def ages_by_cluster(label, customer_age):
    """
    Create a bar plot to compare cluster ages.

    Parameters:
        label (numpy.array): Cluster labels.
        customer_age (pandas.Series): Customer age values.

    Returns:
        None
    """
    sns.barplot(x=label, y=customer_age, estimator=np.mean)
    plt.xlabel('Cluster')
    plt.ylabel('Age')
    plt.title('Comparison of Cluster Ages')
    plt.show()

# ages_by_cluster(label, df['customer_age'])


# In[ ]:


def analyze_distributions(df):
    """
    Analyze the distributions of numerical columns in the DataFrame.

    Parameters:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: Summary statistics of the numerical columns.
    """
    return df.describe()


# In[ ]:


def visualize_relationships(df):
    """
    Create visualizations for different relationships.

    Parameters:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        None
    """
    category_revenue = df.groupby('product_category')['revenue'].sum()
    gender_quantity = df.groupby('customer_gender')['sub_category'].sum()
    gender_age = df.groupby('customer_age')['customer_gender'].sum()

    # Visualize category revenue
    sns.barplot(x=category_revenue.index, y=category_revenue.values)
    plt.xlabel('Product Category')
    plt.ylabel('Revenue')
    plt.title('Total Revenue by Product Category')
    plt.show()

    # Visualize gender quantity
    sns.barplot(x=gender_quantity.index, y=gender_quantity.values)
    plt.xlabel('Customer Gender')
    plt.ylabel('Quantity')
    plt.title('Total Quantity by Customer Gender')
    plt.show()

    # Visualize gender age
    ages_by_cluster(df['customer_gender'], df['customer_age'])
    
# visualize_relationships(df)


# In[ ]:


def visualize_category_revenue(df):
    """
    Create a visualization of total revenue by product sub-category.

    Parameters:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        None
    """
    category_revenue = df.groupby('sub_category')['revenue'].sum()

    sns.barplot(x=category_revenue.index, y=category_revenue.values)
    plt.xlabel('Sub Category')
    plt.ylabel('Revenue')
    plt.title('Total Revenue by Product Sub-Category')

    # Rotate x-labels 90 degrees
    plt.xticks(rotation=90)
    plt.show()

    
# visualize_category_revenue(df)


# In[ ]:


def visualize_gender_quantity(df):
    """
    Create a visualization of total quantity by customer gender.

    Parameters:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        None
    """
    gender_quantity = df.groupby('customer_gender')['sub_category'].sum()
    sns.barplot(x=gender_quantity.index, y=gender_quantity.values)
    plt.xlabel('Customer Gender')
    plt.ylabel('Quantity')
    plt.title('Total Quantity by Customer Gender')
    plt.show()

    
# visualize_gender_quantity(df)


# In[ ]:


def visualize_gender_age(df):
    """
    Create a visualization to compare cluster ages and country of origin.

    Parameters:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        None
    """
    ages_by_cluster(df['customer_gender'], df['customer_age'])
    
# visualize_gender_age(df)

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def visualize_subcategory_country(df):
    """
    Create visualizations of sub-category revenue by country.

    Parameters:
        df (pandas.DataFrame): Input DataFrame containing sub-category and revenue by country.

    Returns:
        None
    """
    # Group the data by sub-category and country and calculate the sum of revenue
    subcategory_country_data = df.groupby(['sub_category', 'country']).agg({'revenue': 'sum'}).reset_index()

    # Visualize sub-category revenue by country
    plt.figure(figsize=(12, 6))
    sns.barplot(data=subcategory_country_data, x='country', y='revenue', hue='sub_category')
    plt.xlabel('Country')
    plt.ylabel('Revenue')
    plt.title('Sub-Category Revenue by Country')
    
    # Display legend outside the chart
    plt.legend(title='Sub-Category', bbox_to_anchor=(1.01, 1), loc='upper left')
    
    plt.show()

# visualize_subcategory_country(df)


import pandas as pd
import matplotlib.pyplot as plt

def visualize_category_profit(category_profit):
    """
    Create a visualization of category profit.

    Parameters:
        category_profit (pandas.Series): Series containing category profit.

    Returns:
        None
    """
    # Convert category_profit to a DataFrame for better display
    category_profit_df = pd.DataFrame(category_profit)
    category_profit_df.columns = ['Profit']


    # Create the bar plot
    plt.figure(figsize=(10, 6))
    category_profit.plot(kind='bar')
    plt.xlabel('Product Category')
    plt.ylabel('Profit')
    plt.title('Category Profit')
    plt.show()

# visualize_category_profit(category_profit)

import matplotlib.pyplot as plt

def visualize_subcategory_profit_revenue(df):
    """
    Create a visualization of sub-category profits and revenue.

    Parameters:
        df (pandas.DataFrame): Input DataFrame with 'sub_category', 'profit', and 'revenue' columns.

    Returns:
        None
    """
    subcategory_profit = df.groupby('sub_category')['profit'].sum()
    subcategory_revenue = df.groupby('sub_category')['revenue'].sum()

    # Create bar plots for sub-category profits and revenue
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    subcategory_profit.plot(kind='bar')
    plt.xlabel('Sub-Category')
    plt.ylabel('Profit')
    plt.title('Sub-Category Profits')

    plt.subplot(2, 1, 2)
    subcategory_revenue.plot(kind='bar')
    plt.xlabel('Sub-Category')
    plt.ylabel('Revenue')
    plt.title('Sub-Category Revenue')

    plt.tight_layout()
    plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt


def visualize_subcategory_revenue(df):
    """
    Create a visualization of revenue for sub-categories: Mountain Bikes, Road Bikes, and Touring Bikes.

    Parameters:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        None
    """
    subcategory_revenue = df[df['sub_category'].isin(['Mountain Bikes', 'Road Bikes', 'Touring Bikes'])]
    subcategory_revenue = subcategory_revenue.groupby('sub_category')['revenue'].sum()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=subcategory_revenue.index, y=subcategory_revenue.values)
    plt.xlabel('Sub-Category')
    plt.ylabel('Revenue')
    plt.title('Revenue by Sub-Category')
    plt.show()

# visualize_subcategory_revenue(df)

def visualize_subcategory_profit(df):
    """
    Create a visualization of profit for sub-categories: Mountain Bikes, Road Bikes, and Touring Bikes.

    Parameters:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        None
    """
    subcategory_profit = df[df['sub_category'].isin(['Mountain Bikes', 'Road Bikes', 'Touring Bikes'])]
    subcategory_profit = subcategory_profit.groupby('sub_category')['profit'].sum()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=subcategory_profit.index, y=subcategory_profit.values)
    plt.xlabel('Sub-Category')
    plt.ylabel('Profit')
    plt.title('Profit by Sub-Category')
    plt.show()

# visualize_subcategory_profit(df)

def visualize_subcategory_gender(df):
    """
    Create a visualization of sub-categories: Mountain Bikes, Road Bikes, and Touring Bikes by gender.

    Parameters:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        None
    """
    subcategory_gender = df[df['sub_category'].isin(['Mountain Bikes', 'Road Bikes', 'Touring Bikes'])]
    subcategory_gender = subcategory_gender.groupby(['sub_category', 'customer_gender']).size().unstack()

    plt.figure(figsize=(10, 6))
    subcategory_gender.plot(kind='bar', stacked=True)
    plt.xlabel('Sub-Category')
    plt.ylabel('Count')
    plt.title('Sub-Category Distribution by Gender')
    plt.legend(title='Gender')
    plt.show()

# visualize_subcategory_gender(df)


def visualize_subcategory_location(df):
    """
    Create a visualization of sub-categories: Mountain Bikes, Road Bikes, and Touring Bikes by location.

    Parameters:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        None
    """
    subcategory_location = df[df['sub_category'].isin(['Mountain Bikes', 'Road Bikes', 'Touring Bikes'])]
    subcategory_location = subcategory_location.groupby(['sub_category', 'country']).size().unstack()

    plt.figure(figsize=(10, 6))
    subcategory_location.plot(kind='bar', stacked=True)
    plt.xlabel('Sub-Category')
    plt.ylabel('Count')
    plt.title('Sub-Category Distribution by Location')
    plt.legend(title='Country')
    plt.show()

# visualize_subcategory_location(df)


def visualize_subcategory_age(df):
    """
    Create a visualization of sub-categories: Mountain Bikes, Road Bikes, and Touring Bikes by age.

    Parameters:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        None
    """
    subcategory_age = df[df['sub_category'].isin(['Mountain Bikes', 'Road Bikes', 'Touring Bikes'])]
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=subcategory_age['sub_category'], y=subcategory_age['customer_age'])
    plt.xlabel('Sub-Category')
    plt.ylabel('Age')
    plt.title('Sub-Category Distribution by Age')
    plt.show()

# visualize_subcategory_age(df)