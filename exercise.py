# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from math import sqrt
dataset = pd.read_csv("bank_customer.csv")
df = pd.DataFrame(dataset)
bank_data = df.copy()

def combine_job_poutcome(bank_data):
    bank_data['job'] = bank_data['job'].replace(['management', 'admin.'], 'white-collar')
    bank_data['job'] = bank_data['job'].replace(['services', 'housemaid'], 'pink-collar')
    bank_data['job'] = bank_data['job'].replace(['retired', 'student','unemployed','unknown'], 'other')

    bank_data['poutcome'] = bank_data['poutcome'].replace(['other','unknown'], 'unknown')
    
combine_job_poutcome(bank_data)    


def convert_categorical_values(bank_data):
    
    le = LabelEncoder()
    bank_data['job'] = le.fit_transform(bank_data['job'])
    bank_data['marital'] = le.fit_transform(bank_data['marital'])
    bank_data['education'] = le.fit_transform(bank_data['education'])
    bank_data['default'] = le.fit_transform(bank_data['default'])
    bank_data['housing'] = le.fit_transform(bank_data['housing'])
    bank_data['loan'] = le.fit_transform(bank_data['loan'])
    bank_data['contact'] = le.fit_transform(bank_data['contact'])
    bank_data['month'] = le.fit_transform(bank_data['month'])
    bank_data['poutcome'] = le.fit_transform(bank_data['poutcome'])
    bank_data['deposit'] = le.fit_transform(bank_data['deposit'])
  
    
convert_categorical_values(bank_data)    

data_1 = bank_data[['age','job','marital','education','balance','housing','duration','poutcome']]
data_2 = bank_data[['job','marital','education','housing']]


def train_and_test(dataset):
    print("Accuracy values of the dataset :")
    X = dataset
    y = bank_data['deposit'].to_frame() 
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=100)
    return X_train,X_test,y_train,y_test

X_train,X_test,y_train,y_test = train_and_test(data_1)

def with_entropy(depth):
    
    tree_with_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth = depth, min_samples_leaf = 5)
    tree_with_entropy.fit(X_train, y_train)
    return tree_with_entropy


def with_gini_index(depth):
    
    tree_with_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth = depth, min_samples_leaf = 5)
    tree_with_gini.fit(X_train, y_train)
    return tree_with_gini


def prediction(X_test, tree):
    
    prediction = tree.predict(X_test)
    print(prediction)
    return prediction
   

def calculate_test_accuracy(tree):
    tree_score_test = tree.score(X_test, y_test)
    print("Test score: ",tree_score_test)
    return tree_score_test

def calculate_train_accuracy(tree):
    tree_score_train = tree.score(X_train, y_train)
    print("Training score: ",tree_score_train)
    return tree_score_train

def calculate_confidence_interval(accuracy,dataset):
    z = 1.96
    interval = z * sqrt((accuracy * (1-accuracy)) / 100)
    return interval
    
def upper_lower_p():
    alpha = 0.5 #because %95
    lower_p = alpha / 2
    upper_p = upper_p = (100 - alpha) + (alpha / 2.0)
    print("lower_p :", lower_p,"upper_p :", upper_p)
    

def plot_tree(tree_name, file_name):
    
    dot_data = StringIO()
    export_graphviz(tree_name, out_file=dot_data,
                    feature_names=X_train.columns, 
                    filled=True, rounded=True,
                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    Image(graph.create_png())

    graph.write_png(file_name)    
    
def display(dataset):    
    print("--------------","DEPTH 3","--------------------")
    gini_tree_depth_3 = with_gini_index(3)    
    
    entropy_tree_depth_3 = with_entropy(3)
    
    gini_prediction = prediction(X_test, gini_tree_depth_3)
    print("Gini Accuracy is :")
    calculate_test_accuracy(gini_tree_depth_3)
    calculate_train_accuracy(gini_tree_depth_3)
    
    entropy_prediction = prediction(X_test, entropy_tree_depth_3)
    print("Entropy Accuracy is :")
    calculate_test_accuracy(entropy_tree_depth_3)
    calculate_train_accuracy(entropy_tree_depth_3)
    
    print("Interval of accuracy: ")
    interval = calculate_confidence_interval(calculate_test_accuracy(gini_tree_depth_3),data_1)
    print(interval)
    
    upper_lower_p()
    
    
    print("--------------","DEPTH 7","--------------------")
    gini_tree_depth_7 = with_gini_index(7)    
    
    entropy_tree_depth_7 = with_entropy(7)
    
    gini_prediction = prediction(X_test, gini_tree_depth_7)
    print("Gini Accuracy is :")
    calculate_test_accuracy(gini_tree_depth_7)
    calculate_train_accuracy(gini_tree_depth_7)
    
    entropy_prediction = prediction(X_test, entropy_tree_depth_7)
    print("Entropy Accuracy is :")
    calculate_test_accuracy(entropy_tree_depth_7)
    calculate_train_accuracy(entropy_tree_depth_7)
    
    print("Interval of accuracy: ")
    interval = calculate_confidence_interval(calculate_test_accuracy(gini_tree_depth_7),data_1)
    print(interval)
    
    upper_lower_p()
    
      
    plot_tree(gini_tree_depth_7, "gini_data1_7.png")    
    plot_tree(entropy_tree_depth_7, "entropy_data1_7.png")      
        
    plot_tree(gini_tree_depth_3, "gini_data1_3.png")    
    plot_tree(entropy_tree_depth_3, "entropy_data1_3.png")   
        
