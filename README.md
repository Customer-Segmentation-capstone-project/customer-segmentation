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

1. What are the distinct characteristics of the four clusters of customers? What is the most influential driver(s) of segmenting customers?
2. On average, is the reveneue of each of the customer segemnts different from one another? Are any of segmented customers spending more? Are there any distinguishable demographic information of customer segments who spend more? What types of products are they buying?
3. Does the average age of the customers among the clusters differ?
4. Are there any sub-categories of products that customers of specific genders are purchasing? What are they?
5. Between road, mountain, and touring bikes, are customers in specific geographical areas buying more of these types of bikes? Does the age differ between the customers who purchase these types of bikes? Do people of different gender purchase one type of bike more than the other?
6. Proportional to a nation's population, does one country purchase more items than the others? What are they buying if so, and how much are they spending?


## Dataset

- The project uses two different datasets. 1) Bicycle Shop Sales data and 2) Amazon product list of cycling gear. 

Link to 1:  https://data.world/vineet/salesdata
Link to 2:  https://www.kaggle.com/datasets/lokeshparab/amazon-products-dataset (you will need a Kaggle account to access)

## Data dictionary
   Dataset 1: Bicycle Sales Data
| **Object Returned** | **Description** |
|:-------------------|:--------------------------------|
| 1. date | date of transaction (datetime) |
| 2. year | year of transaction (object) |
| 3. month | month pf transaction (integer) |
| 4. customer_age |  age of the customer at the time of purchase (Float) |
| 5. customer_gender | gender of customer (female/male) |
| 6. country | country of origin of customer (object) |
| 7. state | state or province of customer (Float)  |
| 8. product_category | broad category of item purchased (object)   |
| 9. sub_category | descriptive category of item purchased(object)   |
| 10. quantity | the number of products purchased for the transaction (Float)  |
| 11. unit_cost | price which the store paid per item purhased in transaction (Float)  |
| 12. unit_price | price which the customer paid per item purchased in transaction (float)   |
| 13. cost | total prie the store paid for the quantity of item purchased in transaction(float)   |
| 14. revenue | total revenue accrued for the quantity  of items purchaseed in transaction (Float)  |

   Dataset 2: Amazon Cycling Product Data
| **Object Returned** | **Description** |
|:-------------------|:--------------------------------|
| 1. | name	The name of the product |
| 2. | main_category	The main category of the product belong |
| 3. | sub_category	The main category of the product belong |
| 4. | image	The image of the product look like |
| 5. | link	The amazon website reference link of the product |
| 6. | ratings	The ratings given by amazon customers of the product |
| 7. | no of ratings	The number of ratings given to this product in amazon shopping |
| 8. | discount_price	The discount prices of the product |
| 9. | actual_price	The actual MRP of the product |

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

