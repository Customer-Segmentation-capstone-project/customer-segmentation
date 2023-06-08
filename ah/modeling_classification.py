from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import wrangle as w

# train, validate, test = w.split_data(w.k_means_clustering(k=4))

def get_modeling_feats_and_target(train, validate, test):
    """
    Perform classification on the given dataset using a decision tree classifier.

    Args:
        df (DataFrame): The input dataset.

    Returns:
        tuple: A tuple containing the trained pipeline, test features, and test labels.
    """
    X_cols = ['customer_age', 'quantity', 'unit_cost',
       'unit_price', 'cost', 'revenue', 'profit', 'sub_category_Bike Stands',
       'sub_category_Bottles and Cages', 'sub_category_Caps',
       'sub_category_Cleaners', 'sub_category_Fenders', 'sub_category_Gloves',
       'sub_category_Helmets', 'sub_category_Hydration Packs',
       'sub_category_Jerseys', 'sub_category_Mountain Bikes',
       'sub_category_Road Bikes', 'sub_category_Shorts', 'sub_category_Socks',
       'sub_category_Tires and Tubes', 'sub_category_Touring Bikes',
       'sub_category_Vests', 'customer_gender_M', 'country_Germany',
       'country_United Kingdom', 'country_United States',
       'product_category_encoded']
    
    # train, validate, test = w.split_data(w.k_means_clustering(k=4))
    
    X_train = train[X_cols]
    X_validate = validate[X_cols]
    X_test = test[X_cols]
    y_train = train['clusters']
    y_validate = validate['clusters']
    y_test = test['clusters']
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test

# X_train, y_train, X_validate, y_validate, X_test, y_test = get_modeling_feats_and_target()

def evaluate_decision_tree(max_depth_range, X_train, y_train, X_validate, y_validate):
    results = []
    
    for max_depth in range(1, max_depth_range + 1):
        # Create decision tree classifier with current max_depth
        dt_classifier = DecisionTreeClassifier(max_depth=max_depth)
        
        # Fit the classifier on the training data
        dt_classifier.fit(X_train, y_train)
        
        # Make predictions on the training set
        y_train_pred = dt_classifier.predict(X_train)
        
        # Calculate evaluation metrics for training set
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred, average='macro')
        train_recall = recall_score(y_train, y_train_pred, average='macro')
        train_f1 = f1_score(y_train, y_train_pred, average='macro')
        
        # Make predictions on the validation set
        y_val_pred = dt_classifier.predict(X_validate)
        
        # Calculate evaluation metrics for validation set
        val_accuracy = accuracy_score(y_validate, y_val_pred)
        val_precision = precision_score(y_validate, y_val_pred, average='macro')
        val_recall = recall_score(y_validate, y_val_pred, average='macro')
        val_f1 = f1_score(y_validate, y_val_pred, average='macro')
        
        # Append results to the list
        results.append([max_depth, train_accuracy, train_precision, train_recall, train_f1,
                        val_accuracy, val_precision, val_recall, val_f1])
    
    # Create a data frame with the evaluation metrics
    dt_evaluation_df = pd.DataFrame(results, columns=['Max Depth', 'Train_Accuracy', 'Train_Precision',
                                                'Train_Recall', 'Train_F1_Score', 'Val_Accuracy',
                                                'Val_Precision', 'Val_Recall', 'Val_F1_Score'])
    
    return dt_evaluation_df

def evaluate_random_forest(X_train, y_train, X_validate, y_validate):
    # Initialize an empty list to store the trained classifiers
    classifiers = []
    
    # Initialize an empty dataframe to store the evaluation metrics
    rf_evaluation_df = pd.DataFrame(columns=['Max Depth', 'Min Leaf', 'Train_Accuracy', 'Train_Precision', 'Train_Recall',
                                             'Train_F1 Score', 'Val_Accuracy', 'Val_Precision', 'Val_Recall',
                                             'Val_F1 Score'])
    
    # Iterate over a range of values for max_depth and min_samples_leaf
    for max_depth in range(1, 10):
        for min_samples_leaf in range(1, 10):
            # Create a random forest classifier with the current max_depth and min_samples_leaf values
            clf = RandomForestClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
            
            # Train the classifier using the given features X and labels y
            clf.fit(X_train, y_train)
            
            # Append the trained classifier to the list
            classifiers.append(clf)
            
            # Make predictions on the training set
            y_train_pred = clf.predict(X_train)
            
            # Calculate train evaluation metrics
            train_accuracy = accuracy_score(y_train, y_train_pred)
            train_precision = precision_score(y_train, y_train_pred, average='macro')
            train_recall = recall_score(y_train, y_train_pred, average='macro')
            train_f1 = f1_score(y_train, y_train_pred, average='macro')
            
            # Make predictions on validation set
            y_val_pred = clf.predict(X_validate)
            
            # Calculate evaluation metrics
            val_accuracy = accuracy_score(y_validate, y_val_pred)
            val_precision = precision_score(y_validate, y_val_pred, average='macro')
            val_recall = recall_score(y_validate, y_val_pred, average='macro')
            val_f1 = f1_score(y_validate, y_val_pred, average='macro')
            
            # Add the evaluation metrics to the dataframe
            rf_evaluation_df = rf_evaluation_df.append({
                'Max Depth': max_depth,
                'Min Leaf': min_samples_leaf,
                'Train_Accuracy': train_accuracy,
                'Train_Precision': train_precision,
                'Train_Recall': train_recall,
                'Train_F1 Score': train_f1,
                'Val_Accuracy': val_accuracy,
                'Val_Precision': val_precision,
                'Val_Recall': val_recall,
                'Val_F1 Score': val_f1
                }, ignore_index=True)
    
    return rf_evaluation_df

def find_rf_best_model_by_recall_difference(X_train, y_train, X_validate, y_validate):

    #Bring in Random Forest evaluation df
    rf_evaluation_df = evaluate_random_forest(X_train, y_train, X_validate, y_validate)

    # Calculate the difference between train recall and validate recall
    rf_evaluation_df['Recall_Difference'] = rf_evaluation_df['Train_Recall'] - rf_evaluation_df['Val_Recall']
    
    # Sort the dataframe by 'Val_Recall' column in descending order
    sorted_df = rf_evaluation_df.sort_values(by='Val_Recall', ascending=False)
    
    # Get the row with the lowest difference between train recall and validate recall
    best_model_row = sorted_df.iloc[sorted_df['Recall_Difference'].abs().idxmin()]
    
    # Extract the max_depth and min_samples_leaf values for the best model
    best_max_depth = int(best_model_row['Max Depth'])
    best_min_samples_leaf = int(best_model_row['Min Leaf'])
    
    # Create the best model using the extracted parameters
    rf_best_model = RandomForestClassifier(max_depth=best_max_depth, min_samples_leaf=best_min_samples_leaf)
    
    return rf_best_model, best_model_row

def find_dt_best_model_by_recall_difference(max_depth_range, X_train, y_train, X_validate, y_validate):

    # Bring in evaluation df for Decision Tree models
    dt_evaluation_df = evaluate_decision_tree(max_depth_range, X_train, y_train, X_validate, y_validate)

    # Calculate the difference between train recall and validate recall
    dt_evaluation_df['Recall_Difference'] = dt_evaluation_df['Train_Recall'] - dt_evaluation_df['Val_Recall']
    
    # Sort the dataframe by 'Val_Recall' column in descending order
    sorted_df = dt_evaluation_df.sort_values(by='Val_Recall', ascending=False)
    
    # Get the row with the lowest difference between train recall and validate recall
    best_model_row = sorted_df.iloc[sorted_df['Recall_Difference'].abs().idxmin()]
    
    # Extract the max_depth and min_samples_leaf values for the best model
    best_max_depth = int(best_model_row['Max Depth'])
    
    # Create the best model using the extracted parameters
    dt_best_model = DecisionTreeClassifier(max_depth=best_max_depth)
    
    return dt_best_model, best_model_row

def test_decision_tree(max_depth, X_train, y_train, X_test, y_test):
    # Create a decision tree classifier with the specified max_depth
    clf = DecisionTreeClassifier(max_depth=max_depth)
    
    # Train the classifier
    clf.fit(X_train, y_train)
    
    # Predict on the train and test sets
    train_predictions = clf.predict(X_train)
    test_predictions = clf.predict(X_test)
    
    # Compute evaluation metrics for train set and test set
    train_accuracy = accuracy_score(y_train, train_predictions)
    train_precision = precision_score(y_train, train_predictions, average='macro')
    train_recall = recall_score(y_train, train_predictions, average='macro')
    train_f1_score = f1_score(y_train, train_predictions, average='macro')
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_precision = precision_score(y_test, test_predictions, average='macro')
    test_recall = recall_score(y_test, test_predictions, average='macro')
    test_f1_score = f1_score(y_test, test_predictions, average='macro')

    # Create a dataframe to store the evaluation metrics
    test_eval_df = pd.DataFrame(columns=['Max_Depth', 'Train_Accuracy', 'Train_Precision', 'Train_Recall',
                                         'Train_F1-Score', 'Test_Accuracy', 'Test_Precision',
                                         'Test_Recall', 'Test_F1-Score'])
    
    # Append the evaluation metrics to the dataframe
    test_eval_df = test_eval_df.append({
        'Max_Depth': max_depth,
        'Train_Accuracy': train_accuracy,
        'Train_Precision': train_precision,
        'Train_Recall': train_recall,
        'Train_F1-Score': train_f1_score,
        'Test_Accuracy': test_accuracy,
        'Test_Precision': test_precision,
        'Test_Recall': test_recall,
        'Test_F1-Score': test_f1_score
    }, ignore_index=True)
    
    return test_eval_df

def find_baseline_and_eval_df(train):

    value_counts = train.clusters.value_counts()

    most_common_cluster = value_counts.idxmax()

    train['baseline'] = most_common_cluster

    y = train.clusters

    # Calculate evaluation metrics
    accuracy = accuracy_score(y, train.baseline)
    precision = precision_score(y, train.baseline, average='macro')
    recall = recall_score(y, train.baseline, average='macro')
    f1 = f1_score(y, train.baseline, average='macro')

    # Create a DataFrame with the evaluation metrics
    eval_dict = {
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1],
    }
    baseline_eval_df = pd.DataFrame(eval_dict)

    return baseline_eval_df
