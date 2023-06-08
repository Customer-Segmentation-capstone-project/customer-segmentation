import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# ========================================================================

def check_hypothesis(p_val, test_stat, α=0.05):
    if p_val < α:
        print('\033[32m========== REJECT THE NULL HYPOTHESIS! ==========\033[0m')
        print(f'\033[35mP-Value:\033[0m {p_val:.8f}')
        print(f'\033[35mtest stat value:\033[0m {test_stat:.8f}')
    else:
        print('\033[31m========== ACCEPT THE NULL HYPOTHESIS! ==========\033[0m')
        print(f'\033[35mP-Value:\033[0m {p_val:.8f}')

# ========================================================================

def show_plot_1(train_clus):
    """
    Plot the mean revenue for each cluster.

    Parameters:
        train_clus (pandas.DataFrame): DataFrame with the clustered data.
        label (numpy.array): Cluster labels.

    Returns:
        None
    """
    # create barplot
    sns.barplot(x=train_clus['clusters'], y=train_clus['revenue'], estimator=np.mean)
    # add axis labels
    plt.xlabel('Cluster', size = 16)
    plt.ylabel('Mean Revenue (Dollars)', size=16)
    # add title
    plt.title('Customers in Cluster 3 Produce\nThe Highest Revenue', size=17)
    # resize ticks
    plt.xticks(size=13)
    plt.yticks(size=13)
    # display plot
    plt.show()

# ========================================================================

def get_test_1(train_clus):
    '''
    This will run an ANOVA test to see if the revenue is the same amongst the different
    clusters
    '''
    # create dfs for each cluster
    clus_0 = train_clus[train_clus.clusters == 0]
    clus_1 = train_clus[train_clus.clusters == 1]
    clus_2 = train_clus[train_clus.clusters == 2]
    clus_3 = train_clus[train_clus.clusters == 3]
    # run an ANOVA test on revenue for each cluster
    f, p = stats.f_oneway(clus_0.revenue, clus_1.revenue, clus_2.revenue, clus_3.revenue)
    # display the hypothesis stats
    check_hypothesis(p, f)

# ========================================================================

def get_avg_cluster_revenue(train_clus):
    print(f'cluster 0 average revenue is: \
{round(train_clus[train_clus.clusters == 0].revenue.mean(),2)}')
    print(f'cluster 1 average revenue is: \
{round(train_clus[train_clus.clusters == 1].revenue.mean(),2)}')
    print(f'cluster 2 average revenue is: \
{round(train_clus[train_clus.clusters == 2].revenue.mean(),2)}')
    print(f'cluster 3 average revenue is: \
{round(train_clus[train_clus.clusters == 3].revenue.mean(),2)}')

# ========================================================================

def show_plot_2(train_clus):
    """
    Create a bar plot to compare cluster ages.

    Parameters:
        train_clus (pandas.Series): clustered training dataset.

    Returns:
        None
    """
    # create barplot of mean age by cluster
    sns.barplot(x=train_clus['clusters'], 
                y=train_clus['customer_age'], 
                estimator=np.mean)
    # add axis labels
    plt.xlabel('Cluster', size=17)
    plt.ylabel('Mean Age of Customers', size=17)
    # add titles
    plt.title('Customers in Cluster 1 Have The\n Highest Average Age', size=18)
    # change tick size
    plt.xticks(size=13)
    plt.yticks(size=13)
    # display plot
    plt.show()

# ========================================================================

def get_test_2(train_clus):
    '''
    This will run an ANOVA test to see if the mean of customer_age is different
    amongst each cluster
    '''
    # create dfs for each cluster
    clus_0 = train_clus[train_clus.clusters == 0]
    clus_1 = train_clus[train_clus.clusters == 1]
    clus_2 = train_clus[train_clus.clusters == 2]
    clus_3 = train_clus[train_clus.clusters == 3]
    # anova test is stats.f_oneway
    f, p = stats.f_oneway(clus_0.customer_age, clus_1.customer_age, 
                          clus_2.customer_age, clus_3.customer_age)
    # display the results of the hypothesis test
    check_hypothesis(p, f)

# ========================================================================

def get_avg_cluster_age(train_clus):
    print(f'cluster 0 average customer_age is: \
{round(train_clus[train_clus.clusters == 0].customer_age.mean(),1)}')
    print(f'cluster 1 average customer_age is: \
{round(train_clus[train_clus.clusters == 1].customer_age.mean(),1)}')
    print(f'cluster 2 average customer_age is: \
{round(train_clus[train_clus.clusters == 2].customer_age.mean(),1)}')
    print(f'cluster 3 average customer_age is: \
{round(train_clus[train_clus.clusters == 3].customer_age.mean(),1)}')

# ========================================================================

def show_plot_3(train_clus):
    """
    Create a visualization of sub-categories

    Parameters:
        train_clus (pandas.DataFrame): Input DataFrame.

    Returns:
        None
    """
    # sort the median customer_age for each sub_category
    meds = train_clus.groupby('sub_category').median().\
        sort_values('customer_age', ascending=False).index.to_list()
    # set figure size
    plt.figure(figsize=(10, 6))
    # create a boxplot
    sns.boxplot(x=train_clus['sub_category'], 
                y=train_clus['customer_age'], 
                order=meds)
    # change axis labels
    plt.xlabel('Sub-Category', size=16)
    plt.ylabel('Age', size=16)
    # set tick size
    plt.xticks(rotation=90, size=13)
    plt.yticks(size=13)
    # add title
    plt.title('Age Is Fairly Evenly Distribute Across Sub-Categories', size=18)
    # display plot
    plt.show()

# ========================================================================

def get_test_3(train_clus):
    '''
    This will run a Kruskall Wallis test to see if there is a correlation
    between age and sub_category of item purchased
    '''
    # create a list of sub_categories
    cat_list = train_clus.sub_category.unique()
    # run the kruskal wallis test
    f, p = stats.kruskal(train_clus[train_clus.sub_category == cat_list[0] ].customer_age,
                     train_clus[train_clus.sub_category == cat_list[1] ].customer_age,
                     train_clus[train_clus.sub_category == cat_list[2] ].customer_age,
                     train_clus[train_clus.sub_category == cat_list[3] ].customer_age,
                     train_clus[train_clus.sub_category == cat_list[4] ].customer_age,
                     train_clus[train_clus.sub_category == cat_list[5] ].customer_age,
                     train_clus[train_clus.sub_category == cat_list[6] ].customer_age,
                     train_clus[train_clus.sub_category == cat_list[7] ].customer_age,
                     train_clus[train_clus.sub_category == cat_list[8] ].customer_age,
                     train_clus[train_clus.sub_category == cat_list[9] ].customer_age,
                     train_clus[train_clus.sub_category == cat_list[10] ].customer_age,
                     train_clus[train_clus.sub_category == cat_list[11] ].customer_age,
                     train_clus[train_clus.sub_category == cat_list[12] ].customer_age,
                     train_clus[train_clus.sub_category == cat_list[13] ].customer_age,
                     train_clus[train_clus.sub_category == cat_list[14] ].customer_age,
                     train_clus[train_clus.sub_category == cat_list[15] ].customer_age,
                     train_clus[train_clus.sub_category == cat_list[16] ].customer_age,
                    )
    # display results of the hypothesis test
    check_hypothesis(p, f)

# ========================================================================

def show_plot_4(train_clus):
    """
    Create a visualization of sub-categories by gender.

    Parameters:
        train_clus (pandas.DataFrame): Input DataFrame.

    Returns:
        None
    """
    # group by sub-category and gender
    subcategory_gender = train_clus.groupby(
        ['sub_category', 'customer_gender']).size().unstack()
    # create plot
    subcategory_gender.plot(kind='bar', stacked=False)
    # add axis labels
    plt.xlabel('Sub-Category', size=16)
    plt.ylabel('Count', size=16)
    # add title
    plt.title('Sub-Category is Evenly Distributed by Gender', size=17)
    # show legend
    plt.legend(title='Gender')
    # display plot
    plt.show()

# ========================================================================    
    
def get_test_4(train_clus):
    '''
    This will run a chi-squared test to see if the sub_category of item purchased
    is dependent on the customer_gender
    '''
    # create a list of sub_categories
    cat_list = train_clus.sub_category.unique()
    # lets make a df of purchase of each sub_category by gender
    # create empty lists for the results
    male=[]
    female=[]
    # cycle through the categories to get a count of items purchased by category
    for i in range(len(cat_list)):
        male.append(train_clus[(train_clus.customer_gender == 'M') & 
                               (train_clus.sub_category == cat_list[i])].\
                                sub_category.count())
        female.append(train_clus[(train_clus.customer_gender == 'F') & 
                               (train_clus.sub_category == cat_list[i])].\
                                sub_category.count())
    # combine the male and female info into one df
    gender_cat = pd.concat([pd.Series(male), pd.Series(female)], axis=1)
    gender_cat = gender_cat.set_index(cat_list).rename(columns={0:'male', 1:'female'})
    # now we can use the gender_cat df in a chi-squared test
    chi2, p, dof, hypothetical = stats.chi2_contingency(gender_cat)
    # display the results of the stats test
    check_hypothesis(p, chi2)

# ========================================================================

def show_plot_5(df):
    """
    Create a visualization of sub-categories: 
    Mountain Bikes, Road Bikes, and Touring Bikes by age.

    Parameters:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        None
    """
    # get a subset of the data containing only bike purchases
    subcategory_age = df[df['sub_category'].\
                         isin(['Mountain Bikes', 'Road Bikes', 'Touring Bikes'])]
    # sort the median customer_age for each sub_category
    meds = subcategory_age.groupby('sub_category').median().\
        sort_values('customer_age', ascending=False).index.to_list()
    # create boxplot
    sns.boxplot(x=subcategory_age['sub_category'], 
                y=subcategory_age['customer_age'],
                order=meds)
    # set axis labels
    plt.xlabel('Bike Type Purchased', size=16)
    plt.ylabel('Age of Customer', size=16)
    # create title
    plt.title('There is a Slight Difference In Age\n Distribution by Bike Type Purchased',
             size=17)
    # change tick size
    plt.xticks(size=13)
    plt.yticks(size=13)
    # display the plot
    plt.show()

# ========================================================================

def get_test_5(train_clus):
    '''
    This will run an ANOVA test to see if customer_age is different amongst customers
    who purchased the different types of bike
    '''
    # run the ANOVA test on bike type and customer age
    f, p = stats.f_oneway(
        train_clus[train_clus.sub_category == 'Road Bikes'].customer_age, 
        train_clus[train_clus.sub_category == 'Mountain Bikes'].customer_age, 
        train_clus[train_clus.sub_category == 'Touring Bikes'].customer_age)
    # display the results of the hypothesis test
    check_hypothesis(p, f)

# ========================================================================
def show_plot_6(df):
    """
    Create a visualization of sub-categories: Mountain Bikes, Road Bikes, 
    and Touring Bikes by gender.

    Parameters:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        None
    """
    # get a subset of the data containing only bike purchases
    subcategory_gender = df[df['sub_category'].\
                            isin(['Mountain Bikes', 'Road Bikes', 'Touring Bikes'])]
    # groupby category and gender
    subcategory_gender = subcategory_gender.\
        groupby(['sub_category', 'customer_gender']).size().unstack()
    # create a plot
    subcategory_gender.plot(kind='bar', stacked=False)
    # change axis labels
    plt.xlabel('Type of Bike Purchased', size=16)
    plt.ylabel('Purchase Count', size=16)
    # add title
    plt.title('Type of Bike Purchased is\n Evenly Distributed By Gender', size=17)
    # show legend
    plt.legend(title='Gender')
    # change tick size
    plt.xticks(rotation=45, size=13)
    plt.yticks(size=13)
    # display the plot
    plt.show()

# ========================================================================

def get_test_6(train_clus):
    '''
    This will run a chi-squared test to see if the type of bike purchased is dependent
    upon customer_gender
    '''
    # create a list of the bike types
    bike_list = ['Road Bikes', 'Mountain Bikes', 'Touring Bikes']
    # lets make a df of bike type purchase by gender
    # create empty lists to store the results
    male=[]
    female=[]
    # cycle through the list of bike types to get a count of purchases by gender
    for i in range(len(bike_list)):
        male.append(train_clus[(train_clus.customer_gender == 'M') & 
                               (train_clus.sub_category == bike_list[i])].\
                                sub_category.count())
        female.append(train_clus[(train_clus.customer_gender == 'F') & 
                               (train_clus.sub_category == bike_list[i])].\
                                sub_category.count())
    # combine male and female results into one df
    gender_bike = pd.concat([pd.Series(male), pd.Series(female)], axis=1)
    gender_bike = gender_bike.rename(columns={0:'male', 1:'female'})\
        .set_index(np.array(bike_list))
    # now we can use the gender_bike df in a chi-squared test
    chi2, p, dof, hypothetical = stats.chi2_contingency(gender_bike)
    # display the results of the hypothesis test
    check_hypothesis(p, chi2)

# ========================================================================

def show_plot_7(df):
    """
    Create a visualization of sub-categories: Mountain Bikes, Road Bikes, and Touring Bikes by location.

    Parameters:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        None
    """
    # create subgroup with only bike purchases
    subcategory_location = df[df['sub_category'].\
                              isin(['Mountain Bikes', 'Road Bikes', 'Touring Bikes'])]
    # group data by sub_category and country
    subcategory_location = subcategory_location.\
        groupby(['sub_category', 'country']).size().unstack()
    # create barplot
    subcategory_location.plot(kind='bar', stacked=False)
    # add axis labels
    plt.xlabel('Type of Bike Purchased', size=16)
    plt.ylabel('Purchase Count', size=16)
    # add title
    plt.title('There Were More Bikes Purchased In\n United States Than Other Countries',
             size=17)
    # show legend
    plt.legend(title='Country')
    # change tick size
    plt.xticks(rotation=45, size=13)
    plt.yticks(size=13)
    # display the plot
    plt.show()

# ========================================================================

def get_test_7(train_clus):
    '''
    This will run a chi-square test to see if the type of bike purchased is dependent
    upon the country of purchase
    '''
    # create a list of the bike types
    bike_list = ['Road Bikes', 'Mountain Bikes', 'Touring Bikes']
    # get a list of countryies in the dataset
    country_list = train_clus.country.unique()
    # create empty lists to store results
    country_a=[]
    country_b=[]
    country_c=[]
    country_d=[]
    # cycle through countries and bike types to get order counts
    for i in range(len(bike_list)):
        country_a.append(train_clus[(train_clus.country == country_list[0]) & 
                               (train_clus.sub_category == bike_list[i])].\
                                sub_category.count())
        country_b.append(train_clus[(train_clus.country == country_list[1]) & 
                               (train_clus.sub_category == bike_list[i])].\
                                sub_category.count())
        country_c.append(train_clus[(train_clus.country == country_list[2]) & 
                               (train_clus.sub_category == bike_list[i])].\
                                sub_category.count())
        country_d.append(train_clus[(train_clus.country == country_list[3]) & 
                               (train_clus.sub_category == bike_list[i])].\
                                sub_category.count())
    # combine results into one df
    country_bike = pd.concat([pd.Series(country_a), 
                              pd.Series(country_b),
                              pd.Series(country_c), 
                              pd.Series(country_d)], axis=1)
    country_bike = country_bike.rename(columns={0:'United Kingdom', 
                                                1:'United States',
                                                2:'France',
                                                3:'Germany'})\
                            .set_index(np.array(bike_list))
    
    # now we can use the gender_bike df in a chi-squared test
    chi2, p, dof, hypothetical = stats.chi2_contingency(country_bike)
    # display hypothesis test results
    check_hypothesis(p, chi2)

# ========================================================================

def show_plot_8(train_clus):
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
    # sort the values by proportion
    purchase_prop = purchase_prop.sort_values('proportion', ascending=False)
    
    # create the plot
    sns.barplot(x=purchase_prop.country, y=purchase_prop.proportion)
    # add title
    plt.title('United Kingdom Has The Highest Purchase Rate\n\
    Relative to Total Country Population', size=17)
    # add axis labels
    plt.xlabel('Country', size=16)
    plt.ylabel('Order Proportion to Total Population', size=15)
    # change tick size
    plt.xticks(rotation=45, size=13)
    plt.yticks(size=13)
    # display tthe plot
    plt.show()

# ========================================================================

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

# ========================================================================
    
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

# ========================================================================

def product_rec_wordcloud(cluster_words_series, cluster_name):

    img = WordCloud(background_color='White', colormap='Set2'
         ).generate(' '.join(cluster_words_series))
    
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Most Words for {cluster_name}')
    plt.show()