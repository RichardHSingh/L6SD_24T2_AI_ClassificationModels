import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn


# =========== Model Libraries ===============
# Package needed --> pip install scikit-learn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from mpl_toolkits.mplot3d import Axes3D
import joblib


# =========== AI Models ===============
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report



# =========== IMPORTING DATA & SPLITTING ===============
from sklearn.datasets import load_iris

iris = load_iris()
iris_x = iris.data
iris_y = iris.target




# ============= CONVERTING IRIS TO DATAFRAME ===========
df_iris = pd.DataFrame(data = iris.data, columns = iris.feature_names)




# ============= Display first 5 rows of the dataset ===========
print(df_iris.head())
# or
print(df_iris[:5])


# creating space from content to content
print("\n\n")




# ============= Display last 5 rows of the dataset ============
print(df_iris.tail(5))
# or 
print(df_iris[-15:])


# creating space from content to content
print("\n\n")




# ============= Determine shape of the dataset (shape - total numbers of rows and columns) ============
iris_shape = df_iris.shape
print(f"\nShape of given dataframe {iris_shape}. To be precise, this dataset consists of {iris_shape[0]} rows and {iris_shape[1]} columns\n")

# creating space from content to content
print("\n\n")




# ============= Display concise summary of the dataset (info) ============
print(f"\nConcise summary of given data set is {df_iris.info()}\n")

# creating space from content to content
print("\n\n")





# ============= Check the null values in dataset (isnull) ============
# isnull will show nulls
# sum will show count of the null values
print(f"\nTotal ammount of null values in data set is: \n{df_iris.isnull().sum()}\n")

# creating space from content to content
print("\n\n")




# ============ Identify library to plot graph to understand relations among various columns =============
# installing seaborn library pip install seaborn
# imported matplotlib
# shows relationship between data
# no need to print as show does it

# Pair plot
print("Relations between given data columns")
sbn.pairplot(df_iris)
plt.show()

# creating space from content to content
print("\n\n")




# ============ Create input dataset from original dataset by dropping irrelevant features ============
# I feel like given 4 columns, none are irrelevant in the dataset but ...

X = df_iris




# ============ Create output dataset from original dataset =============
Y = pd.Series(iris_y)





# ============ Transform input dataset into percentage based weighted between 0 and 1 =============
scale = MinMaxScaler()
x_scaler = scale.fit_transform(X)
print(x_scaler)

# creating space from content to content
print("\n\n")




# ============ Transform output dataset into percentage based weighted between 0 and 1 =============

sc = MinMaxScaler()
y_reshape = Y.values.reshape(-1,1)
y_scaler = sc.fit_transform(y_reshape)
print(y_scaler)

# creating space from content to content
print("\n\n")

#===== OR =====

# cause y is catergoical
y_scaled = Y / Y.max()
print(y_scaled)

# creating space from content to content
print("\n\n")






# ============ Print first few rows of scaled input dataset =============
# :5 number is based on what is written in array so if it says :10, 10 rows of data will show
print(f"\nthe follow is the data from dropped columns \n{x_scaler[:5]}")

# creating space from content to content
print("\n\n")





# ============ Print first few rows of scaled output dataset =============
# :5 number is based on what is written in array so if it says :8, 8 rows of data will show
print(f"\nthe follow is the data from output columns \n {y_scaler[:7]}")

# creating space from content to content
print("\n\n")







# ============ Split data into training and testing sets =============
# split the dataset 25/75
# if the size has been dictated, shuffle wouldnt take affect
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.25, train_size = 0.75, random_state = 42, shuffle = True)

print("X_train results:\n", X_train)
print("X_test results:\n", X_test)
print("Y_train results:\n", Y_train)
print("Y_test results:\n", Y_test)

# creating space from content to content
print("\n\n")





# ============ Print shape of test and training data =============
print("Shape of training given dateset:\n", X_train.shape)
print("Shape of test given dateset", X_test.shape)

# creating space from content to content
print("\n")

print("Shape of training given dateset:", Y_train.shape)
print("Shape of test given dateset\n", Y_test.shape)


# creating space from content to content
print("\n\n")




# ============ Print first few rows of test and training data =============
print("First few rows of training given dateset:\n", X_train[:5])
print("First few rows of test given dateset", X_test[:6])

# creating space from content to content
print("\n")

print("First few rows of training given dateset:", Y_train[:7])
print("First few rows of test given dateset\n", Y_test[:8])

# creating space from content to content
print("\n\n")





# ============ Import and initialize AI models (need 10) =============
# models used [LogisticRegression | RandomForestClassifier]

# assign variables

logistic_model = LogisticRegression()
randForest_model = RandomForestClassifier(n_estimators = 100, random_state = 42)
desTree_model = DecisionTreeClassifier()





# ============ Train models using training data =============
# use .fit -->  method is how a machine learning model learns from the training data. It adjusts the model's parameters so that it can make accurate predictions.

# assigned variables
logistic_model.fit(X_train, Y_train)
randForest_model.fit(X_train, Y_train)
desTree_model.fit(X_train, Y_train)







# ============ Prediction on test data =============
# predicting the x_test  to the models variables from earlier

# assigned variables
logistic_pred = logistic_model.predict(X_test)
randForest_pred = randForest_model.predict(X_test)
desTree_pred = desTree_model.predict(X_test)




# ============ Prediction on test data =============
logistic_accuracy = accuracy_score(Y_test, logistic_pred)
randForect_accuracy = accuracy_score(Y_test, randForest_pred)
desTree_accuracy = accuracy_score(Y_test, desTree_pred)

logistic_report = classification_report(Y_test, logistic_pred, target_names = iris.target_names)
randForest_report = classification_report(Y_test, randForest_pred, target_names = iris.target_names)
desTree_report = classification_report(Y_test, randForest_pred, target_names = iris.target_names)


print(f"First accuracy is:\n {logistic_accuracy}\n")
print(f"First classification report based on Logistic Classification:\n {logistic_report }")

print("\n\n")

print(f"Second accuracy is:\n {randForect_accuracy}\n")
print(f"Second classification report based on Random Forest Classification:\n {randForest_report}")

print("\n\n")

print(f"Second accuracy is:\n {desTree_accuracy}\n")
print(f"Second classification report based on Random Forest Classification:\n {desTree_report}")
