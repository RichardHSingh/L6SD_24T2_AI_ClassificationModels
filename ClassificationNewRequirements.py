import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn


# =========== Model Libraries ===============
# Package needed --> pip install scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib


# =========== AI Models ===============
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# =========== USER INTERFACE ===============
import tkinter as tk
from tkinter import messagebox


# =========== IMPORTING | READING DATA & SPLITTING ===============
# from sklearn.datasets import load_iris

df_iris = pd.read_csv("iris.csv")





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

X = df_iris.drop(["variety"], axis = 1)
print(f"Dropped coloumn from iris dataset is: {X}")


# creating space from content to content
print("\n\n")




# ============ Create output dataset from original dataset =============
Y = df_iris["variety"]
print(f"Output coloumn from iris dataset is: {Y}")


# creating space from content to content
print("\n\n")





# ============ Split data into training and testing sets =============
stand_scaler = StandardScaler()
standardised_x = stand_scaler.fit_transform(X)

print(f"Standard dataset is:\n {standardised_x[:4]}")


# creating space from content to content
print("\n\n")









# ============ Print first few rows of scaled input dataset =============
# :5 number is based on what is written in array so if it says :10, 10 rows of data will show
print(f"\nThe following is the data from dropped columns \n{standardised_x[:5]}")

# creating space from content to content
print("\n\n")






# ============ Split data into training and testing sets =============
# split the dataset 25/75
# if the size has been dictated, shuffle wouldnt take affect
X_train, X_test, Y_train, Y_test = train_test_split( standardised_x, Y, test_size = 0.25, train_size = 0.75, random_state = 42, shuffle = True)

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
randForest_accuracy = accuracy_score(Y_test, randForest_pred)
desTree_accuracy = accuracy_score(Y_test, desTree_pred)

logistic_report = classification_report(Y_test, logistic_pred)
randForest_report = classification_report(Y_test, randForest_pred)
desTree_report = classification_report(Y_test, randForest_pred)


print(f"First accuracy is:\n {logistic_accuracy}\n")
print(f"First classification report based on Logistic Classification:\n {logistic_report }")

print("\n\n")

print(f"Second accuracy is:\n {randForest_accuracy}\n")
print(f"Second classification report based on Random Forest Classification:\n {randForest_report}")

print("\n\n")

print(f"Second accuracy is:\n {desTree_accuracy}\n")
print(f"Second classification report based on Random Forest Classification:\n {desTree_report}")



# creating space from content to content
print("\n\n")




# ============ Creating dictionary for models & Find accurate model =============
# creating dictionary with model and key value
accuracy_model = {
    "Logistic Regression": logistic_accuracy,
    "Random Forest Classifier": randForest_accuracy,
    "Desision Tree Classifer": desTree_accuracy
}


better_performing_model = max(accuracy_model, key = accuracy_model.get)
elite_model_performance = None

# seeing which model is most accurate through branch coverage
if better_performing_model == "Logistic Regression":
    elite_model_performance = logistic_model
elif better_performing_model == "Random Forest Classifier":
    elite_model_performance = randForest_model
else:
    elite_model_performance == desTree_model

print(f"Better or most accurate model is: {better_performing_model}")
print(f"Accuracy with this model is: {accuracy_model[better_performing_model]}")


# saving the accurate model --> extension joblib
joblib.dump(elite_model_performance, "elite_model.joblib")


# loading the saved model
load_model = joblib.load("elite_model.joblib")


# creating space from content to content
print("\n\n")




# # ============ User Inputs ============
# sepal_length = float(input("Please enter the length of the sepal in cm: "))
# sepal_width = float(input("Please enter the width of the sepal in cm: "))
# petal_length = float(input("Please enter the length of the petal in cm: "))
# petal_width = float(input("Please enter the width of the petal in cm: "))


# print("\n\n")



# user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
# user_input_scaling = stand_scaler.transform(user_input)
# user_input_prediction = load_model.predict(user_input_scaling)

# print(f"Prediction for user's entered data is: {user_input}. The related flower to input is: {user_input_prediction[0]}")


# ===============================================================================================================================================================================
# ========================================================================= USER INTERFACE ======================================================================================
# ===============================================================================================================================================================================
def flower_prediction():
    try:
        # ============ User input field ============
        # getting users input from UI entry fields
        sepal_length = float(entry_sepal_length.get())
        sepal_width = float(entry_sepal_width.get())
        petal_length = float(entry_petal_length.get())
        petal_width = float(entry_petal_width.get())


        # ============ Predicting using given inputs ============
        user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        user_input_scaled = stand_scaler.transform(user_input)
        prediction = load_model.predict(user_input_scaled)
        

        # ============ Predicting Output ============
        messagebox.showinfo("Prediction", f"The predicted flower type is: {prediction[0]}")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values")

# ============================================
# ============ CREATING UI WINDOW ============
window = tk.Tk()
window.title("Iris Flower Prediction")

# =============================================
# ============ ADDING INPUT FIELDS ============
tk.Label(window, text = "Please enter the length of the sepal in cm: ").grid(row = 0, column = 0)
entry_sepal_length = tk.Entry(window)
entry_sepal_length.grid(row = 0, column = 1, padx = 10, pady = 5)

tk.Label(window, text = "Please enter the width of the sepal in cm: ").grid(row = 1, column = 0)
entry_sepal_width = tk.Entry(window)
entry_sepal_width.grid(row = 1, column = 1, padx = 10, pady = 5)

tk.Label(window, text = "Please enter the length of the petal in cm:").grid(row = 2, column = 0)
entry_petal_length = tk.Entry(window)
entry_petal_length.grid(row = 2, column = 1, padx = 10, pady = 5)

tk.Label(window, text = "Please enter the width of the petal in cm:").grid(row = 3, column = 0)
entry_petal_width = tk.Entry(window)
entry_petal_width.grid(row = 3, column = 1, padx = 10, pady = 5)

# Create and place the prediction button
predict_button = tk.Button(window, text = "Prediction", command = flower_prediction)
predict_button.grid(row = 5, columnspan = 2, padx = 10, pady = 5)

# Run the Tkinter main loop
window.mainloop()

# script to run
# python ClassificationNewRequirements.py