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
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from sklearn import tree
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.tree import export_graphviz

import warnings
warnings.filterwarnings(action='ignore')

#Other additional imports
warnings.filterwarnings('ignore') 

# %%
df = pd.read_csv("spam.csv")

# %%
df.info()

# %%
plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
plt.pie(df['Class'].value_counts(),labels=['Ham','Spam'],autopct="%0.2f")
plt.subplot(1,2,2)
sns.barplot(x=df['Class'].value_counts().index,y=df['Class'].value_counts(),data=df)
plt.show()
plt.savefig('Ham_Spam.png',format='png',bbox_inches = "tight")

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
print ("Prediction:",dt_predict,'\n')
print("Accuracy:", round((accuracy_score(y_test, dt_predict)),4), "\n")
print("Report: \n", classification_report(y_test, dt_predict)) 
print("Confusion Matrix: \n", confusion_matrix(y_test, dt_predict), "\n")
plot_confusion_matrix(decision_tree_clf, x_test, y_test)
plt.title("Decision tree Confusion Matrix")
plt.savefig('DC_CM.png',format='png',bbox_inches = "tight")

# %%
n_nodes = decision_tree_clf.tree_.node_count
children_left = decision_tree_clf.tree_.children_left
children_right = decision_tree_clf.tree_.children_right
feature = decision_tree_clf.tree_.feature
threshold = decision_tree_clf.tree_.threshold

node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
while len(stack) > 0:
    # `pop` ensures each node is only visited once
    node_id, depth = stack.pop()
    node_depth[node_id] = depth

    # If the left and right child of a node is not the same we have a split
    # node
    is_split_node = children_left[node_id] != children_right[node_id]
    # If a split node, append left and right children and depth to `stack`
    # so we can loop through them
    if is_split_node:
        stack.append((children_left[node_id], depth + 1))
        stack.append((children_right[node_id], depth + 1))
    else:
        is_leaves[node_id] = True

print(
    "The binary tree structure has {n} nodes and has "
    "the following tree structure:\n".format(n=n_nodes)
)
for i in range(n_nodes):
    if is_leaves[i]:
        print(
            "{space}node={node} is a leaf node.".format(
                space=node_depth[i] * "\t", node=i
            )
        )
    else:
        print(
            "{space}node={node} is a split node: "
            "go to node {left} if X[:, {feature}] <= {threshold} "
            "else to node {right}.".format(
                space=node_depth[i] * "\t",
                node=i,
                left=children_left[i],
                feature=feature[i],
                threshold=threshold[i],
                right=children_right[i],
            )
        )

# %%
#plt.figure(figsize=(25,25))
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (5,5), dpi=2000)
tree.plot_tree(decision_tree_clf, filled=True)
plt.title("Decision tree")
plt.show()
plt.savefig('Dtree.jpg',format='jpg')
plt.savefig('filename.png')

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
print('Random Forest Classification model','\n')
for i in features: #for loop for feature selection
    print('Random Forest model with Feature:',i,'\n')
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

print("Random Forest classifier with SQRT and 5000  estimators\n")
print ("Prediction:",rf_predict,'\n')
print("Accuracy:", round((accuracy_score(Y_test, rf_predict)),4), "\n")
print("Report: \n", classification_report(Y_test, rf_predict)) 
print("Confusion Matrix: \n", confusion_matrix(Y_test, rf_predict), "\n")
plot_confusion_matrix(random_forest_clf, x_test, y_test)
plt.title("Random Forest Confusion Matrix")
plt.savefig('RF_CM.png',format='png',bbox_inches = "tight")

# %%
fn=X_train.columns
cn= 'class'
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (5,5), dpi=2000)
tree.plot_tree(random_forest_clf.estimators_[50],
               feature_names = fn, 
               class_names=cn,
               filled = True)
plt.title("Random Forest Tree")
plt.savefig('RFtree.png',format='png',bbox_inches = "tight")

# %%



