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


## Clustered Data Dictionary

| Sub Category        | Encoded |
|---------------------|---------|
| Bike Racks          | 0       |
| Bike Stands         | 1       |
| Bottles and Cages   | 2       |
| Caps                | 3       |
| Cleaners            | 4       |
| Fenders             | 5       |
| Gloves              | 6       |
| Helmets             | 7       |
| Hydration Packs     | 8       |
| Jerseys             | 9       |
| Mountain Bikes      | 10      |
| Road Bikes          | 11      |
| Shorts              | 12      |
| Socks               | 13      |
| Tires and Tubes     | 14      |
| Touring Bikes       | 15      |
| Vests               | 16      |

=======
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
>>>>>>> dbd0365d4cea245ce0225fd971ece8604861515b

## Setup

To download the dataset, you will need an account for [Kaggle.com](https://www.kaggle.com).

- Installs needed to run this notebook:
    - python -m pip install --upgrade wordcloud
- Gives instructions for reproducing your work. i.e. Running your notebook on someone else's computer.
- List the required Python libraries and their versions.
- Include instructions for setting up a virtual environment, if necessary.
- Provide any additional setup instructions, if needed.

## Data Preprocessing

- Project Plan Guides the reader through the different stages of the pipeline as they relate to your project
- Briefly describe the data preprocessing steps, including handling missing values, encoding categorical variables, scaling or normalizing numerical variables, and feature engineering.

The data will be downloaded from Kaggle and stored on our local devices.

## Feature Engineering

- In our project, we implemented a clustering operation on our dataset. The purpose of this clustering is to discover inherent groupings in the data. This operation involves several steps, each of which contributes to the transformation and processing of our data.

### Data Standardization

- Before running the clustering algorithm, the dataset must be scaled to level the playing field for the dataset. Most importanlty, when dealing with features that are measured in different units, the data muyst be scaled. If one feature's range is significantly larger than the others, it could dominate the others in the clustering algorithm, which relies on distance metrics. To solve this, we standardize our features using sklearn's `StandardScaler`. This subtracts the mean and scales the data to unit variance.

### Clustering Algorithm

- With properly standardized data, we use the K-means clustering algorithm from sklearn to cluster the dataset. K-means is a widely-used method in cluster analysis. It partitions n observations into k clusters, where each observation belongs to the cluster with the nearest mean. We choose k=4, representing the number of clusters we want to partition our data into. The choice of k depends on the data and should be chosen after careful consideration. We then fit and predict our standardized data using the K-means model to get our labels.

### Adding Cluster Labels to DataFrame

- Once we obtain our cluster labels, we add them to our original DataFrame as a new column named 'cluster'.


## Model Selection, Training, and Validation

### Model Description

- Employed a Decision Tree Classifier model from the sklearn library. A decision tree is a flowchart-like tree structure where an internal node represents feature (or attribute), the branch represents a decision rule, and each leaf node represents the outcome.

- Desicion Trees suit our use case due to its interpretability and effectiveness with categorical features. The target variable, `cluster`, is a categorical variable that we're trying to predict based on several other features in our dataset.

### Model Selection Process

- The selection process for this model was based on the types of data and the problems we were trying to solve. Decision Tree's serve as excellent models for classification problems. THe Decision Tree Model accomodates the projects need to understand and visuaalize data which make it a number once chioice due to interpretability.

### Model Training and Validation

- Modeling was straightforward, we separated our dataset into features (`X`) and the target variable (`y`). The features considered are, 'quantity', 'unit_cost', 'unit_price', 'cost', 'revenue'. The target variable is the 'cluster'.

- We then split the data into training, validation, and testing sets using sklearn's `train_test_split` function, with xx% of the data going to the training set, xx% to the validation setr, and xx% to the test set.

- We used training data to train our Decision Tree Classifier, fitted the model to our training data using the `fit` method, which is a part of the sklearn library's API.

- After model training, we validated the model performance by making it predict the `cluster` for our test data. We then compare these predictions to the actual labels using a classification report, which provides an in-depth analysis of the model's performance, including precision, recall, f1-score, and support for each class, and accuracy of the model.

- In this implementation, we have not used hyperparameter tuning or cross-validation. However, in a more complex setup, these could be valuable tools to optimize model performance. 

<!-- ![Model Performance Comparison](path/to/model_performance_image.png) -->

*Image caption: A comparison of the performance of different models on the dataset.*

## Results

- Summarize the project results, including the best-performing model, its performance metrics, and any insights derived from the analysis.

## Future Work/Next Steps:

- Employ GridSearchCV for hyperparameter tuning, which would involve training the model multiple times with different combinations of hyperparameters to find the best performing set. 

## Acknowledgements

- List any references, articles, or resources used during the project.
- Acknowledge any collaborators or external support, if applicable.
https://www.canva.com/design/DAFkl7WSndQ/LvgJcc-wH_v0ry8fTSGzSA/edit
