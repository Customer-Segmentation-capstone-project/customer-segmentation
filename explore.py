#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import seaborn as sns
<<<<<<< HEAD
import matplotlib.pyplot as plt
=======
from wordcloud import WordCloud
>>>>>>> d5ff9c7b869a049c17176438b06bb9f37150b789

def plot_mean_revenue(clustered_df, label):
    """
    Plot the mean revenue for each cluster.

    Parameters:
        clustered_df (pandas.DataFrame): DataFrame with the clustered data.
        label (numpy.array): Cluster labels.

    Returns:
        None
    """
<<<<<<< HEAD
    sns.barplot(x=label, y=clustered_df['revenue'], estimator=np.mean)
    plt.xlabel('Cluster')
    plt.ylabel('Mean Revenue')
    plt.title('Mean Revenue for Each Cluster')
    plt.show()
    
# run line below to execute function    
# plot_mean_revenue(clustered_df, label)

def mean_revenue_viz(clustered_df, label):
    """
    Plot the mean revenue for each cluster.

    Parameters:
        clustered_df (pandas.DataFrame): DataFrame with the clustered data.
        label (numpy.array): Cluster labels.

    Returns:
        None
    """
    cluster_labels = ['Essential Cyclist', 'All-Rounder Cyclist', 'Sporty Cyclist', 'Bike Enthusiast']
    sns.barplot(x=label, y=clustered_df['revenue'], estimator=np.mean)
    plt.xlabel('Cluster')
    plt.ylabel('Mean Revenue')
    plt.title('Mean Revenue for Each Cluster')
    plt.xticks(range(len(cluster_labels)), cluster_labels)  # Set the x-axis tick labels to cluster labels
=======
    # create barplot
    sns.barplot(x=train_clus['clusters'], y=train_clus['revenue'], estimator=np.mean)
    # add axis labels
    plt.xlabel('Cluster', size = 16)
    plt.ylabel('Mean Revenue (Dollars)', size=16)
    # add title
    plt.title('Customers in Cluster 3 Produce\nThe Highest Average Revenue', size=17)
    # resize ticks
    plt.xticks(size=13)
    plt.yticks(size=13)
    # display plot
>>>>>>> d5ff9c7b869a049c17176438b06bb9f37150b789
    plt.show()


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

<<<<<<< HEAD
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
=======
def show_plot_2(train_clus):
>>>>>>> d5ff9c7b869a049c17176438b06bb9f37150b789
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
<<<<<<< HEAD
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

    # Select the top 5 sub-categories based on revenue
    top_subcategories = subcategory_country_data.groupby('sub_category')['revenue'].sum().nlargest(5).index
    subcategory_country_data = subcategory_country_data[subcategory_country_data['sub_category'].isin(top_subcategories)]

    # Visualize sub-category revenue by country
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=subcategory_country_data, x='country', y='revenue', hue='sub_category')
    plt.xlabel('Country')
    plt.ylabel('Revenue ($)')
    plt.title('Sub-Category Revenue by Country')
    plt.legend(title='Sub-Category', loc='upper left')

    # Formatting y-axis tick labels as dollar amounts
    ax.yaxis.set_major_formatter('${x:,.0f}')

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
=======
    # sort the median customer_age for each sub_category
    meds = train_clus.groupby('sub_category').median().\
        sort_values('customer_age', ascending=False).index.to_list()
    # set figure size
>>>>>>> d5ff9c7b869a049c17176438b06bb9f37150b789
    plt.figure(figsize=(10, 6))
    category_profit.plot(kind='bar')
    plt.xlabel('Product Category')
    plt.ylabel('Profit')
    plt.title('Category Profit')
    plt.show()

# visualize_category_profit(category_profit)

import matplotlib.pyplot as plt
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

<<<<<<< HEAD

def visualize_category_revenue(df):
    """
    Create a visualization of total revenue by product sub-category.

    Parameters:
        df (pandas.DataFrame): Input DataFrame.
=======
def get_test_8(train_clus):
    '''
    This will run a kruskal wallis test to see if the proportion of orders to 
    total population differs amongst countries in the dataset.
    '''
    # create a list of the country total populations as of 2023
    pop_totals = [65_690_000, 83_310_000, 67_620_000, 334_230_000]
    # create a dataframe of orders by country
    purchase_prop = pd.DataFrame(train_clus.groupby('country').
             date.count()).rename(columns={'date':'orders'}).reset_index()
    # create a column with the total populations
    purchase_prop = pd.concat([purchase_prop, pd.Series(pop_totals)], axis=1).\
                    rename(columns={0:'population'})
    # create a column with the proportion of orders to population
    purchase_prop['proportion'] = purchase_prop.orders / purchase_prop.population
    # run a kruskal wallis test
    f, p = stats.kruskal(
                     purchase_prop[purchase_prop.country == 'United Kingdom'].proportion,
                     purchase_prop[purchase_prop.country == 'United States'].proportion,
                     purchase_prop[purchase_prop.country == 'France'].proportion,
                     purchase_prop[purchase_prop.country == 'Germany'].proportion)
    # display the results of the test
    check_hypothesis(p, f)

def most_buying_power(train):
    """
    Visualize the cluster with the most buying power based on revenue.

    Parameters:
        label (numpy.array): Cluster labels.
        revenue (pandas.Series): Revenue values.
>>>>>>> d5ff9c7b869a049c17176438b06bb9f37150b789

    Returns:
        None
    """
<<<<<<< HEAD
    category_revenue = df.groupby('sub_category')['revenue'].sum()

    # Selecting top five sub-categories based on revenue
    top_subcategories = category_revenue.nlargest(5).index
    top_revenue = category_revenue.loc[top_subcategories]

    # Sorting the data in ascending order
    top_revenue_sorted = top_revenue.sort_values(ascending=True)

    # Creating a horizontal bar plot
    ax = sns.barplot(x=top_revenue_sorted.values, y=top_revenue_sorted.index)

    # Updating labels accordingly
    plt.xlabel('Revenue ($)')
    plt.ylabel('Sub Category')
    plt.title('Total Revenue by Top Five Product Sub-Categories')

    # Formatting x-axis tick labels as dollar amounts
    ax.ticklabel_format(style='plain', axis='x')

    plt.show()
    
def visualize_subcategory_profit_revenue(df):
    
    subcategory_profit = df.groupby('sub_category')['profit'].sum()
    subcategory_revenue = df.groupby('sub_category')['revenue'].sum()

    # Selects the top 5 sub-categories based on revenue
    top_subcategories = subcategory_revenue.nlargest(5).index
    subcategory_profit = subcategory_profit[top_subcategories]
    subcategory_revenue = subcategory_revenue[top_subcategories]

    # Sorts the data in ascending order
    subcategory_profit_sorted = subcategory_profit.sort_values()
    subcategory_revenue_sorted = subcategory_revenue.sort_values()

    # Creates bar plots for sub-category profits and revenue
    plt.figure(figsize=(11, 8))
    plt.subplot(2, 1, 1)
    subcategory_profit_sorted.plot(kind='bar')
    plt.xlabel('Sub-Category')
    plt.ylabel('Profit')
    plt.title('Sub-Category Profits')
    plt.xticks(rotation=0)

    # Formatting y-axis tick labels as dollar amounts
    plt.gca().yaxis.set_major_formatter('${x:,.0f}')

    plt.subplot(2, 1, 2)
    subcategory_revenue_sorted.plot(kind='bar')
    plt.xlabel('Sub-Category')
    plt.ylabel('Revenue')
    plt.title('Sub-Category Revenue')
    plt.xticks(rotation=0)

    plt.tight_layout()

    # Formatting y-axis tick labels as dollar amounts
    plt.gca().yaxis.set_major_formatter('${x:,.0f}')

    plt.show()
    
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def visualize_customer_spending(df):
    """
    Create a bar plot to visualize customer spending by age range.

    Parameters:
        df (pandas.DataFrame): Input DataFrame with 'customer_age' and 'cost' columns.

    Returns:
        None
    """
    # Bin the customer ages into ranges
    age_bins = [17, 27, 37, 47, 57, 67, 77, 87, 90]
    age_labels = ['17-26', '27-36', '37-46', '47-56', '57-66', '67-76', '77-86', '87+']
    df['age_range'] = pd.cut(df['customer_age'], bins=age_bins, labels=age_labels)

    # Calculate the total spending for each age range
    age_spending = df.groupby('age_range')['cost'].sum()

    # Create a bar plot of customer spending by age range
    sns.barplot(x=age_spending.index, y=age_spending.values)
    plt.xlabel('Age Range')
    plt.ylabel('Total Spending')
    plt.title('Customer Spending by Age')

    # Format y-axis tick labels as dollar amounts
    formatter = ticker.StrMethodFormatter('${x:,.0f}')
    plt.gca().yaxis.set_major_formatter(formatter)

=======
    sns.countplot(x=train['clusters'])
    plt.xlabel('Cluster', size = 16)
    plt.ylabel('Count of Transactions', size = 16)

    plt.xticks(size=13)
    plt.yticks(size=13)

    plt.title('Distribution of Transaction Counts\nCluster 0 Most Transactions', size=17)
    plt.show()

def viz_age_dis_boxplot(train_clus):
    
    # create the plot
    sns.boxplot(data=train_clus, x=train_clus['clusters'], y=train_clus['customer_age'])
    # add title
    plt.title('Age Distribution by Cluster', size=17)
    # add axis labels
    plt.xlabel('Cluster', size=16)
    plt.ylabel('Age', size=15)
    # change tick size
    plt.xticks(size=13)
    plt.yticks(size=13)
    # display tthe plot
    plt.show()

def split_series_words(dataframe, series_name):
    # Get the series from the DataFrame
    series = dataframe[series_name]
    
    # Split the words in the series
    words_list = series.str.split().tolist()
    
    # Flatten the list of lists into a single list
    flattened_list = [word for sublist in words_list for word in sublist]
    
    # Convert the list into a new Series
    new_series = pd.Series(flattened_list)
    
    return new_series

def product_rec_wordcloud(cluster_words_series, cluster_name):

    img = WordCloud(background_color='White', colormap='Set2'
         ).generate(' '.join(cluster_words_series))
    
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Most Words for {cluster_name}')
>>>>>>> d5ff9c7b869a049c17176438b06bb9f37150b789
    plt.show()