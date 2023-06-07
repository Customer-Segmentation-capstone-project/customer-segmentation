import numpy as np
import pandas as pd
import wrangle_products as w
import re
import nltk

# =====================================================================

def get_products():
    '''
    This will get the amazon products dataset using the wrangle_products file
    
    returns product dataframe
    '''
    # acquire and prepare the amazon products dataset from kaggle or local cached file
    return w.wrangle_products()

# =====================================================================

def get_category_list():
    '''
    This will take the list of sub_categories in our bike shop dataset, 
    it will clean, tokenize, lemmatize and remove stopwords from the category names,
    and combine each sub_category with a regular expression to match against the 
    amazon product names.
    
    Returns a dataframe of sub_category, lemmatized_sub_category, and regexp
    '''
    
    # create a list of the sub_categories from the bike shop, manually gathered
    sub_cat_list = ['Tires and Tubes', 'Gloves', 'Helmets', 'Bike Stands',
                   'Mountain Bikes', 'Hydration Packs', 'Jerseys', 'Fenders',
                   'Cleaners', 'Socks', 'Caps', 'Touring Bikes', 'Bottles and Cages',
                   'Vests', 'Road Bikes', 'Bike Racks', 'Shorts']
    # clean, tokenize, lematize and remove stopwords (including 'bike')
    cat_list = w.get_cat_list(sub_cat_list, extra_words=['bike', 'bikes'])
    # creaate regexp to find the category names within product names
    regex_list = [r'.*tire.*|.*tube.*',
                  r'.*glove.*',
                  r'.*helmet.*',
                  r'.*stand.*',
                  r'.*mountain.*',
                  r'.*hydration.*',
                  r'.*jersey.*',
                  r'.*fender.*',
                  r'.*cleaner.*',
                  r'.*sock.*',
                  r'.*cap.*',
                  r'.*touring.*',
                  r'.*bottle.*|.*cage.*',
                  r'.*vest.*',
                  r'.*road.*',
                  r'.*rack.*',
                  r'.*short.*']
    # combine the items into one dataframe
    cat_df = pd.concat([
                pd.Series(sub_cat_list), 
                pd.Series(cat_list),
                pd.Series(regex_list)
                ], axis=1).\
                rename(columns={0:'shop_cat', 
                                1:'lemmed_cat',
                                2:'regexp'})
    # return the category dataframe
    return cat_df

# =====================================================================

def get_cluster_product_rec(cluster):
    '''
    this will return a list of products from the amazon products dataset,
    based on the passed cluster number generated
    '''
    # get a dataframe of categories
    cat_df = get_category_list()
    # get a dataframe of amazon products
    products = get_products()
    
    # create dictionary to store matches
    cat_matches = {}
    # cycle through all the products in the product df
    for i, product in enumerate(products.name_preped):
        # create an empty list to store category matches for current product
        matching_cats = []
        # cycle through the regexp functions of categories we are looking to match to
        for j, cat in enumerate(cat_df.regexp):
            # check for match with current product to current category regex
            if re.search(cat, product):
                # if there is a match, add category name to match list
                matching_cats.append(cat_df.shop_cat[j])
            # add matches for this product to dict of product matches
            cat_matches[i] = matching_cats
    
    # add product category matches to products dataframe
    products = pd.concat([products, pd.Series(cat_matches)], axis=1).\
        rename(columns={0:'cat_matches'})
    
    # if passed cluster is 0
    if cluster == 0:
        # create a list of the most frequent sub_categories purchased in cluster 0
        clus_0_cats = ['Tires and Tubes', 'Bottles and Cages', 
                       'Caps', 'Helmets', 'Cleaners']
        # get a list of products less than 20 dollars
        group_0 = products[(products.actual_price <20)]
        # sort the products by product rating
        group_0 = group_0.sort_values('prod_rating', ascending=False).\
                reset_index().drop(columns='index')
        # create an empty list to store product matches to common sub_categories purchased
        # by this customer cluster
        clus_matches = []
        # cycle through the products matches for this product group
        for i, product in enumerate(group_0.cat_matches):
            # cycle the name/category matches on this product
            for match in product:
                # check if any name/category matches for this product are what are common
                # for cluster 0 customers
                if match in clus_0_cats:
                    # add the product to list of products to recomend for this cluster
                    clus_matches.append(i)
        # get a subset of products that match our most common sub_category purchased
        # by this cluster
        clus_0_products = group_0[group_0.index.isin(clus_matches)]
        # return the products to recommend to customers in this cluster
        return clus_0_products

    # if passed cluster is 1
    elif cluster == 1:
        # create a list of the most frequent sub_categories purchased in cluster
        clus_1_cats = ['Tires and Tubes', 'Bottles and Cages', 
                       'Helmets', 'Caps', 'Jerseys']
        # get products between 20 to 40 dollars
        group_1 = products[(products.actual_price >= 20) & (products.actual_price < 40)]
        # sort the products by product rating
        group_1 = group_1.sort_values('prod_rating', ascending=False).\
                reset_index().drop(columns='index')
        # create an empty list to store product matches to common sub_categories purchased
        # by this customer cluster
        clus_matches = []
        # cycle through the products matches for this product group
        for i, product in enumerate(group_1.cat_matches):
            # cycle the name/category matches on this product
            for match in product:
                # check if any name/category matches for this product are what are common
                # for cluster customers
                if match in clus_1_cats:
                    # add the product to list of products to recomend for this cluster
                    clus_matches.append(i)
        # get a subset of products that match our most common sub_category purchased
        # by this cluster
        clus_1_products = group_1[group_1.index.isin(clus_matches)]
        # return the products to recommend to customers in this cluster
        return clus_1_products

    # if passed cluster is 1
    elif cluster == 2:
        # create a list of the most frequent sub_categories purchased in cluster
        clus_2_cats = ['Helmets', 'Road Bikes', 'Jerseys', 
                       'Tires and Tubes', 'Mountain Bikes']
        # get products between 40 to 100 dollars
        group_2 = products[(products.actual_price >= 40) & (products.actual_price < 100)]
        # sort the products by product rating
        group_2 = group_2.sort_values('prod_rating', ascending=False).\
                reset_index().drop(columns='index')
        # create an empty list to store product matches to common sub_categories purchased
        # by this customer cluster
        clus_matches = []
        # cycle through the products matches for this product group
        for i, product in enumerate(group_2.cat_matches):
            # cycle the name/category matches on this product
            for match in product:
                # check if any name/category matches for this product are what are common
                # for cluster customers
                if match in clus_2_cats:
                    # add the product to list of products to recomend for this cluster
                    clus_matches.append(i)
        # get a subset of products that match our most common sub_category purchased
        # by this cluster
        clus_2_products = group_2[group_2.index.isin(clus_matches)]
        # return the products to recommend to customers in this cluster
        return clus_2_products
    
    # if passed cluster is 3
    elif cluster == 3:
        # create a list of the most frequent sub_categories purchased in cluster
        clus_3_cats = ['Mountain Bikes', 'Road Bikes', 
                       'Touring Bikes', 'Jerseys', 'Shorts']
        # get products above 100 dollars
        group_3 = products[(products.actual_price >= 100)]
        # sort the products by product rating
        group_3 = group_3.sort_values('prod_rating', ascending=False).\
                reset_index().drop(columns='index')
        # create an empty list to store product matches to common sub_categories purchased
        # by this customer cluster
        clus_matches = []
        # cycle through the products matches for this product group
        for i, product in enumerate(group_3.cat_matches):
            # cycle the name/category matches on this product
            for match in product:
                # check if any name/category matches for this product are what are common
                # for cluster customers
                if match in clus_3_cats:
                    # add the product to list of products to recomend for this cluster
                    clus_matches.append(i)
        # get a subset of products that match our most common sub_category purchased
        # by this cluster
        clus_3_products = group_3[group_3.index.isin(clus_matches)]
        # return the products to recommend to customers in this cluster
        return clus_3_products