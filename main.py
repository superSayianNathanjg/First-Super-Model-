import numpy as np
import pandas as pd
import tensorflow as tf  # This is for neural network.
from sklearn.model_selection import train_test_split  # For splitting dataset into 4 parts.
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Churn_Modelling.csv")

# Analyse dataset. # Check if features are not valuable, then drop.
# print(df)  # Prints entire dataset.
# print(df.head(3))  # Prints 3 rows and x columns.
# print(df.isnull().sum())  #  Check if any columns have missing values.

pd.set_option('display.max_columns', len(df.columns))  # Modify so it displays all columns.

# print(df["Geography"].unique())  # Checking how many unique values there are. Ex, only 3 countries.

# Modifying Dataset
gender = {"Male": 0, "Female": 1}  # Dict for replacing values in the dataframe.
df.Gender = [gender[g] for g in df.Gender]
counties = {"France": 0, "Germany": 2, "Spain": 1}  # Converting unstructured data to numerical "structured" data.
df.Geography = [counties[x] for x in df.Geography]
# print(df[["Geography", "Gender"]])

x = df.iloc[:, 3:-1].values  # From column index 3 to -1 last index. Excluding last index.
y = df.iloc[:, -1].values  # Get last column. "Exited".

""" Split into test and train data """
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.79, random_state=0)

""" Scaling Dataset for ANN - Make it non-linear/Convert to 0 and 1. """
sc = StandardScaler()  # Variable for conducting Standard Scaler.
x_train = sc.fit_transform(x_train)  # Transforms x_train. Between -1 and 1.
x_test = sc.transform(x_test)  # Do not use fit_transform. Transform all test data. So individual test values!!!

""" Initialising ANN """
# Input layer, hidden layer total = 2.
super_model = tf.keras.models.Sequential()  # Start. This is the input layer.
super_model.add(tf.keras.layers.Dense(units=6, activation="relu"))  # Hidden layer 1.
super_model.add(tf.keras.layers.Dense(units=6, activation="relu"))  # Hidden layer 2.
super_model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))  # Output layer.

# Model structure is finished. Now compile.
super_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

""" Training the model """
super_model.fit(x_train, y_train, batch_size=32, epochs=100)

""" Predictions """
# single_value_pred = sc.transform([[600, 0, 0, 23, 4, 6000, 2, 1, 1, 38190]])  # must be 2d array
pred = super_model.predict(x_test)
print(pred)
for x in pred:
    if x > 0.5:
        print("This will leave")
    else:
        print("They will stay")
# if super_model.predict(single_value_pred) > 0.5:
#     print("\nThis guy will leave the bank in six months")
#     print(super_model.predict(single_value_pred))
# else:
#     print('\nThis guy will NOT leave the bank in six months')
#     print(super_model.predict(single_value_pred))
