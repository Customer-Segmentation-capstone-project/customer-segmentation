# SegmentSavvy: Revolutionizing the Shopping Experience for Bicycle Enthusiasts

A Codeup capstone project created by: 
 - Adam Harris
 - Edward Michaud 
 - Caroline Miller
 - Dionne Taylor

<!-- ![Project Banner](path/to/banner_image.png) -->

<!-- *Project banner image credits: [Source Name](image_source_url)* -->

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Setup](#setup)
- [Data Preprocessing](#data-preprocessing)
- [Model Selection and Training](#model-selection-and-training)
- [Results](#results)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)

## Overview

ShopSense is an exciting data-driven project aimed at enhancing the shopping experience for customers of bike shops by providing personalized product recommendations. As a team of data scientists, our goal is to improve customer satisfaction and increase sales by leveraging historical purchasing data to deliver tailored recommendations to each individual customer.

Through advanced machine learning techniques and recommendation algorithms, we will analyze the purchasing history of bike shop customers and identify patterns and preferences. By understanding each customer's unique preferences and needs, we can offer personalized recommendations that align with their interests, resulting in a more enjoyable and efficient shopping experience.

Our approach involves applying recommendation systems to the vast product catalog available on Amazon.com. We will develop a recommendation engine that utilizes customer purchase history, demographic information, and browsing behavior to generate accurate and relevant product suggestions. By harnessing the power of data, we aim to provide highly targeted recommendations that align with each customer's specific interests and needs.

### Project Goal:

The endstate for this project includes three components:  1) Clustering Model which identifies like-minded customers based off of their purchasing power, historical item purchases, and customer age, 2) Classification Model which can identify the cluster in which a customer's transaction can be segmented, and 3) Recommender System which can link the cluster of a customer with a Amazon product which fits the customer's potential interests and budget. The project acts as proof of concept which can be summarized as using consumer sales data for particular categories of goods, in this case cycling equipment, from smaller businesses to feed marketing decisions for a global, ecommerce company like Amazon which sells a vast scope of products. 

### Project Description: 

The project combines two overarching concepts and technologies: 1) customer segementation and 2) product recommendation. The first concept uses a dataset from a small cycling retailor which serves online customers from the United States, the United Kingdom, France, and Germany. The cycling shop sells primarily bikes, accessories, and clothing. Clustering is used to compile similar transactions based on purchasing power (amount spent), category of item purchased, and customer age. These clusters of transactions serve to illustrate distinct customer types. Classification machine learning is then used to cluster transactions based on demographic data as well as transactional data. The second concept uses a dataset which acts as an inventory list for cyling products which Amazon offers. Natural Language Processing techniques to further categorize the cycling product list to fit the clusters of customer types. The customer types from clustering coalesce with the sub-categories of Amazon cycling products in a recommender system which matches appropriate Amazon products to customers based on their interests, past purchases, and inferred budget.

### Project Purpose:

Online consumers use Amazon to buy affordable, convenient products of a wide variety. Amazon makes online shopping easy and expedient. A wide variety of products can prove to be overwhelming for consumers especially if the consumer is looking for a specific product. There are multiple sports, hobbies, and jobs that require specialized expertise and personizalization when choosing products. Some of these activities include mountain climbing, cycling, backpacking, fishing, and equestrianism. People who engage in these kinds of activities are highly skilled and particular about the brands and products they use. The question this project aims to answer is can an online retailor with an immense assortment of products create a boutique style of personalized product recommendations to customers whose purchases revolve around niche and exclusive themes such as cycling? With artificial intelligence and machine learning, personalized customer segementation can be used to tailor online commerse to accomplish this goal.

## Initial Hypotheses Questions

1. Can tailored recommendations improve the average order value by suggesting complementary products?
2. Does providing personalized product recommendations based on purchasing history lead to higher customer engagement?


## Dataset

- DataFrame with 36 columns capturing various aspects of sales data. The columns contain information about sales attributes, customer details, product details, costs, revenues, profits, and encoded categorical variables. The DataFrame represents a structured and organized form of the sales dataset, facilitating data analysis and manipulation taskEntries: The DataFrame contains a total of 34866 entries, meaning there are 34866 rows in the DataFrame.


## Link to datasource: https://data.world/vineet/salesdata

## Data dictionary

Column Name	Description
Year	The year in which the sales were recorded.
Month	The month in which the sales were recorded.
Customer ID	A unique identifier for each customer.
Customer Name	The name of the customer.
Product ID	A unique identifier for each product.
Product Name	The name of the product.
Quantity	The quantity of the product sold.
Revenue	The revenue generated from the sales of the product.
Cost	The cost associated with the product.
Profit	The profit obtained from the sales of the product.
Country	The country where the sales occurred.
State	The state where the sales occurred.
City	The city where the sales occurred.
Region	The region where the sales occurred.
Order Date	The date when the order was placed.
Ship Date	The date when the product was shipped.
Ship Mode	The mode of shipment for the product.
Segment	The market segment to which the customer belongs.
Ship Priority	The priority of the shipment.


## Setup

To download the dataset, you will need an account for [Kaggle.com](https://www.kaggle.com).

- Gives instructions for reproducing your work. i.e. Running your notebook on someone else's computer.
- List the required Python libraries and their versions.
- Include instructions for setting up a virtual environment, if necessary.
- Provide any additional setup instructions, if needed.

## Data Preprocessing

- Project Plan Guides the reader through the different stages of the pipeline as they relate to your project
- Briefly describe the data preprocessing steps, including handling missing values, encoding categorical variables, scaling or normalizing numerical variables, and feature engineering.

The data will be downloaded from Kaggle and stored on our local devices.


## Model Selection and Training

- List the machine learning models considered for the project.
- Explain the model selection process and criteria.
- Describe the model training process, including hyperparameter tuning and cross-validation, if applicable.

<!-- ![Model Performance Comparison](path/to/model_performance_image.png) -->

*Image caption: A comparison of the performance of different models on the dataset.*

## Results

- Summarize the project results, including the best-performing model, its performance metrics, and any insights derived from the analysis.

## Future Work

- Discuss potential improvements, additional features, or alternative approaches for the project.

## Acknowledgements

- List any references, articles, or resources used during the project.
- Acknowledge any collaborators or external support, if applicable.

