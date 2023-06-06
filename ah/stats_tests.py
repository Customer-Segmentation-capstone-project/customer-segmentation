import numpy as np
import pandas as pd
from scipy import stats

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
        male.append(train_clus[(train_clus.customer_gender_M == 1) & 
                               (train_clus.sub_category == cat_list[i])].\
                                sub_category.count())
        female.append(train_clus[(train_clus.customer_gender_M == 0) & 
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
        male.append(train_clus[(train_clus.customer_gender_M == 1) & 
                               (train_clus.sub_category == bike_list[i])].\
                                sub_category.count())
        female.append(train_clus[(train_clus.customer_gender_M == 0) & 
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