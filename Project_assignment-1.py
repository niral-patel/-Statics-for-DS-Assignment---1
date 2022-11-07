# %% [markdown]
# # Project Assignmnet 1 
# ***
# #### Name = NIRAL PATEL
# #### ID = 1303276
# #### Date = 10/17/2022
# #### Description = Decision Tree and Random Forest Classification model
# ***
# ## Reporting Tasks:
# 1. Compare the accuracies of the Random Forest classifier as a function of the number of base learners (e.g., 10, 50, 100, 500, 1000, and 500) and the number of features to consider at each split (e.g., auto or sqrt). Report your observations/conclusions and provide evidence to support your conclusions. [50 points]
# 2. Compare of the results of all the classifiers (with the best possible parameter setting for each classifier). Use classification accuracy (# of instances correctly classified/total # of instances presented for classification), per class classification accuracy, and confusion matrix to compare the classifiers. [50 points]
# 

# %%
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings(action='ignore')

#Other additional imports
warnings.filterwarnings('ignore') 

# %%
df = pd.read_csv("spam.csv")

# %%
df.info()

# %% [markdown]
# #### Train the classifiers using the first 1000 instances and use the remaining 3601 for testing. Feel free to create separate training and testing data files. 
# #### Have your own strategy to deal with any missing feature values (e.g., remove instances with missing features or fill in the missing feature values with the most popular value.).

# %%
# Creatae the Test and Train Data using basic python sysntex 
train_df = df.iloc[:1000,:]
test_df = df.iloc[1000:,:]


X_train = train_df.drop(columns='Class')
Y_train = train_df['Class']
X_test = test_df.drop(columns='Class')
Y_test = test_df['Class']
print("Total Train Data", train_df.shape,'\nTotal Test Data',test_df.shape,"\nTest Data set x:",X_test.shape,'\nTest Data set y:',Y_test.shape,'\nTrain Data set x:',X_train.shape,'\nTrain data set y:',Y_train.shape)

# %%
# Creatae the Test and Train Data using model selection
#--------------------------------
from sklearn.model_selection import train_test_split
#--------------------------------
X = df.drop(columns='Class')
y = df['Class']
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.782)
print("Total Data", df.shape,"\nTest Data set x:",x_test.shape,'\nTest Data set y:',y_test.shape,'\nTrain Data set x:',x_train.shape,'\nTrain data set y:',y_train.shape)

# %%
# Decision Tree classifier
#--------------------------
from sklearn.tree import DecisionTreeClassifier
#--------------------------

# Fit the training model to the desired classifier
decision_tree_clf = DecisionTreeClassifier()
decision_tree_clf = decision_tree_clf.fit(x_train, y_train)
dt_accuracy = decision_tree_clf.score(x_test, y_test)
print("Classification Accuracy: ", round((dt_accuracy),4), "\n")

# %%
# Metrics - Decision Tree classifier
dt_predict = decision_tree_clf.predict(x_test)
print("Decision Tree classifier\n")
print ("Prediction from Decision tree clf:",dt_predict,'\n')
print("Classification Accuracy: \n", round((accuracy_score(y_test, dt_predict)),4), "\n")
print("Classification Report: \n", classification_report(y_test, dt_predict)) 
print("Confusion Matrix: \n", confusion_matrix(y_test, dt_predict), "\n")

# %% [markdown]
# #### 1. Compare the accuracies of the Random Forest classifier as a function of the number of base learners (e.g., 10, 50, 100, 500, 1000, and 500) and the number of features to consider at each split (e.g., auto or sqrt). Report your observations/conclusions and provide evidence to support your conclusions. [50 points]

# %%
# Random Forest classifier
#--------------------------
from sklearn.ensemble import RandomForestClassifier
#--------------------------

estimator = [10,50,100,500,1000,5000] #estimate values
features = ['auto', 'sqrt'] #feature values
rf_accuracy_auto = [] # Empty array to store the Accuracy score
rf_accuracy_sqrt = []
print('\nRandom Forest Classification model','\n')
for i in features: #for loop for feature selection
    print('\nRandom Forest model with Feature:',i,'\n')
    for j in estimator: #for loop for estimator selection
        random_forest_clf = RandomForestClassifier(n_estimators=j, max_features=i) #create a RF model with feature and estimator
        random_forest_clf = random_forest_clf.fit(X_train, Y_train) #Train the model
        rf_score = random_forest_clf.score(X_test, Y_test) # Test the model and get the accuracy score
        print('Classification Accuracy with ',j,' estimators:', round((rf_score),4), "\n")
        if i == 'sqrt':
            rf_accuracy_sqrt.append(rf_score) #store the accuracy score if feature is squrt
        else:
            rf_accuracy_auto.append(rf_score) #store the accuracy score if feature is auto

accuracy1 = pd.DataFrame(rf_accuracy_sqrt,columns = ['accuracy_sqrt'])
accuracy2 = pd.DataFrame(rf_accuracy_auto,columns = ['accuracy_auto'])
accuracy = pd.merge(accuracy1,accuracy2,left_index=True,right_index=True)
print (accuracy)

# %%
accuracy.describe()

# %%
# Metrics - Random Forest classifier 
rf_predict = random_forest_clf.predict(X_test)

print("Random Forest classifier with SQRT\n")
print ("Prediction from Random Forest clf:",rf_predict,'\n')
print("Classification Accuracy: \n", round((accuracy_score(Y_test, rf_predict)),4), "\n")
print("Classification Report: \n", classification_report(Y_test, rf_predict)) 
print("Confusion Matrix: \n", confusion_matrix(Y_test, rf_predict), "\n")



