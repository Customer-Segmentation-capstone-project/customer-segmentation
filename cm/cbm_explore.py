import pandas as pd
import numpy as np
# visualizations
import matplotlib.pyplot as plt
import seaborn as sns
import wrangle as w
import clustering_feature_engineering as c

df, train, validate, test = w.wrangle_data()

train, validate, test = c.engineer_clusters(train, validate, test, k=5)

explore_cols = ['date', 'year', 'month', 'customer_age', 'customer_gender', 'country',
       'state', 'product_category', 'sub_category', 'quantity', 'unit_cost',
       'unit_price', 'cost', 'revenue', 'total_price', 'clusters']

cluster_0 = train[train.clusters == 0][explore_cols]
cluster_1 = train[train.clusters == 1][explore_cols]
cluster_2 = train[train.clusters == 2][explore_cols]
cluster_3 = train[train.clusters == 3][explore_cols]
cluster_4 = train[train.clusters == 4][explore_cols]

clusters = [cluster_0, cluster_1, cluster_2, cluster_3, cluster_4]

def viz_distribution_age():

    fig, axs = plt.subplots(len(clusters), 1, figsize=(15, 30))

    axs[0].hist(cluster_0.customer_age, color='cadetblue', ec='black')
    axs[1].hist(cluster_1.customer_age, color='darkturquoise', ec='black')
    axs[2].hist(cluster_2.customer_age, color='skyblue', ec='black')
    axs[3].hist(cluster_3.customer_age, color='teal', ec='black')
    axs[4].hist(cluster_4.customer_age, color='mediumaquamarine', ec='black')

    fig.suptitle("Comparing the Distribution of Customer Age Across Clusters")
    axs[0].set_title('Cluster 0 Customer Age Distribution')
    axs[1].set_title('Cluster 1 Customer Age Distribution')
    axs[2].set_title('Cluster 2 Customer Age Distribution')
    axs[3].set_title('Cluster 3 Customer Age Distribution')
    axs[4].set_title('Cluster 4 Customer Age Distribution')

    axs[0].set_xlabel('Age Brackets')
    axs[0].set_ylabel('Count')
    axs[1].set_xlabel('Age Brackets')
    axs[1].set_ylabel('Count')
    axs[2].set_xlabel('Age Brackets')
    axs[2].set_ylabel('Count')
    axs[3].set_xlabel('Age Brackets')
    axs[3].set_ylabel('Count')
    axs[4].set_xlabel('Age Brackets')
    axs[4].set_ylabel('Count')

    plt.show()

def get_count_gender():
    ''' Gets boxplots of acquired continuous variables'''

    plt.figure(figsize=(5, 25))

    for i, cluster in enumerate(clusters):
        
        plot_number = i + 1 
        
        # Create subplot.
        plt.subplot(len(clusters), 1, plot_number)

        # Title with column name.
        plt.title(f'Comparing Counts of Transactions by Gender for cluster_{i}')

        # Display boxplot for column.
        sns.countplot(x=cluster['customer_gender'], hue=cluster['product_category'], palette='crest', ec='black')

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Hide gridlines.
        plt.grid(False)

    plt.show()

def get_count_country():
    ''' Gets boxplots of acquired continuous variables'''

    plt.figure(figsize=(5, 25))

    for i, cluster in enumerate(clusters):
        
        plot_number = i + 1 
        
        # Create subplot.
        plt.subplot(len(clusters), 1, plot_number)

        # Title with column name.
        plt.title(f'Comparing Counts of Transactions by Country for cluster_{i}')

        # Display boxplot for column.
        sns.countplot(x=cluster['country'], hue=cluster['product_category'], palette='crest', ec='black')

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Hide gridlines.
        plt.grid(False)

    plt.show()