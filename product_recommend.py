import numpy as np
import pandas as pd
import wrangle_products as w
import re
import nltk

def get_products():
    return w.wrangle_products()

def get_category_list():
    sub_cat_list = ['Tires and Tubes', 'Gloves', 'Helmets', 'Bike Stands',
                   'Mountain Bikes', 'Hydration Packs', 'Jerseys', 'Fenders',
                   'Cleaners', 'Socks', 'Caps', 'Touring Bikes', 'Bottles and Cages',
                   'Vests', 'Road Bikes', 'Bike Racks', 'Shorts']
    cat_list = w.get_cat_list(sub_cat_list, extra_words=['bike', 'bikes'])
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
    cat_df = pd.concat([
                pd.Series(sub_cat_list), 
                pd.Series(cat_list),
                pd.Series(regex_list)
                ], axis=1).\
                rename(columns={0:'shop_cat', 
                                1:'lemmed_cat',
                                2:'regexp'})
    
    return cat_df

def get_cluster_product_rec(cluster):
    
    cat_df = get_category_list()
    products = get_products()
    
    cat_matches = {}
    for i, product in enumerate(products.name_preped):
        matching_cats = []
        for j, cat in enumerate(cat_df.regexp):
            if re.search(cat, product):
                matching_cats.append(cat_df.shop_cat[j])
            cat_matches[i] = matching_cats

    products = pd.concat([products, pd.Series(cat_matches)], axis=1).\
        rename(columns={0:'cat_matches'})
    
    if cluster == 0:
        clus_0_cats = ['Tires and Tubes', 'Bottles and Cages', 
                       'Caps', 'Helmets', 'Cleaners']
        group_0 = products[(products.actual_price <20)]
        group_0 = group_0.sort_values('prod_rating', ascending=False).\
                reset_index().drop(columns='index')
        clus_matches = []
        for i, product in enumerate(group_0.cat_matches):
            for match in product:
                if match in clus_0_cats:
                    clus_matches.append(i)
        clus_0_products = group_0[group_0.index.isin(clus_matches)]
        return clus_0_products

    elif cluster == 1:
        clus_1_cats = ['Tires and Tubes', 'Bottles and Cages', 
                       'Helmets', 'Caps', 'Jerseys']
        group_1 = products[(products.actual_price >= 20) & (products.actual_price < 40)]
        group_1 = group_1.sort_values('prod_rating', ascending=False).\
                reset_index().drop(columns='index')
        clus_matches = []
        for i, product in enumerate(group_1.cat_matches):
            for match in product:
                if match in clus_1_cats:
                    clus_matches.append(i)
        clus_1_products = group_1[group_1.index.isin(clus_matches)]
        return clus_1_products

    elif cluster == 2:
        clus_2_cats = ['Helmets', 'Road Bikes', 'Jerseys', 
                       'Tires and Tubes', 'Mountain Bikes']
        group_2 = products[(products.actual_price >= 40) & (products.actual_price < 100)]
        group_2 = group_2.sort_values('prod_rating', ascending=False).\
                reset_index().drop(columns='index')
        clus_matches = []
        for i, product in enumerate(group_2.cat_matches):
            for match in product:
                if match in clus_2_cats:
                    clus_matches.append(i)
        clus_2_products = group_2[group_2.index.isin(clus_matches)]
        return clus_2_products

    elif cluster == 3:
        clus_3_cats = ['Mountain Bikes', 'Road Bikes', 
                       'Touring Bikes', 'Jerseys', 'Shorts']
        group_3 = products[(products.actual_price >= 100)]
        group_3 = group_3.sort_values('prod_rating', ascending=False).\
                reset_index().drop(columns='index')
        clus_matches = []
        for i, product in enumerate(group_3.cat_matches):
            for match in product:
                if match in clus_3_cats:
                    clus_matches.append(i)
        clus_3_products = group_3[group_3.index.isin(clus_matches)]
        return clus_3_products