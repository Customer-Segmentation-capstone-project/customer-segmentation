# ShopSense: Revolutionizing the Shopping Experience for Bicycle Enthusiasts

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

## Project Goals

1. Improve customer satisfaction: By providing personalized recommendations, we aim to enhance the shopping experience and help customers discover products that resonate with their interests and preferences.
2. Increase sales: Tailored recommendations have the potential to boost customer engagement and drive incremental sales by showcasing relevant products to customers who are more likely to make a purchase.
3. Foster customer loyalty: By demonstrating a deep understanding of each customer's preferences, we aim to build strong customer relationships and foster loyalty within the bike shop community.

## Initial Hypotheses Questions

1. Can tailored recommendations improve the average order value by suggesting complementary products?
2. Does providing personalized product recommendations based on purchasing history lead to higher customer engagement?


## Dataset

- DataFrame with 36 columns capturing various aspects of sales data. The columns contain information about sales attributes, customer details, product details, costs, revenues, profits, and encoded categorical variables. The DataFrame represents a structured and organized form of the sales dataset, facilitating data analysis and manipulation taskEntries: The DataFrame contains a total of 34866 entries, meaning there are 34866 rows in the DataFrame.


## Link to datasource: https://data.world/vineet/salesdata

## Data dictionary

| Column Name               | Description                                              |
| ------------------------- | -------------------------------------------------------- |
| year                      | The year in which the sales were recorded.               |
| month                     | The month in which the sales were recorded.              |
| customer_age              | The age of the customer.                                 |
| customer_gender           | The gender of the customer.                              |
| country                   | The country where the sales occurred.                    |
| state                     | The state where the sales occurred.                      |
| product_category          | The category of the product.                             |
| sub_category              | The sub-category of the product.                         |
| quantity                  | The quantity of the product sold.                        |
| unit_cost                 | The cost per unit of the product.                        |
| ...                       | ...                                                      |
| sub_category_Shorts       | Binary indicator (0 or 1) for the sub-category "Shorts". |
| sub_category_Socks        | Binary indicator (0 or 1) for the sub-category "Socks".  |
| sub_category_Tires        | Binary indicator (0 or 1) for the sub-category "Tires".  |
| sub_category_Touring Bikes| Binary indicator (0 or 1) for the sub-category "Touring Bikes".|
| sub_category_Vests        | Binary indicator (0 or 1) for the sub-category "Vests".  |
| customer_gender_M         | Binary indicator (0 or 1) for male customers.            |
| country_Germany           | Binary indicator (0 or 1) for sales in Germany.          |
| country_United Kingdom    | Binary indicator (0 or 1) for sales in the United Kingdom. |
| country_United States     | Binary indicator (0 or 1) for sales in the United States.|
| product_category_encoded  | Numeric encoding for the product category.               |


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

