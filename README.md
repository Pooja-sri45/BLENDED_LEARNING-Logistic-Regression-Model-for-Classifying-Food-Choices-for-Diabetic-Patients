# BLENDED_LEARNING
# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Libraries: Import necessary libraries such as pandas, numpy, matplotlib, and sklearn.
2. Load Dataset: Load the dataset containing car prices and relevant features.
3. Data Preprocessing: Handle missing values and perform feature selection if necessary.
4. Split Data: Split the dataset into training and testing sets.
5. Train Model: Create a linear regression model and fit it to the training data.
6. Make Predictions: Use the model to make predictions on the test set.
7. Evaluate Model: Assess model performance using metrics like R² score, Mean Absolute Error (MAE), etc.
8. Check Assumptions: Plot residuals to check for homoscedasticity, normality, and linearity.
9. Output Results: Display the predictions and evaluation metrics.


## Program:
```
Program to implement Logistic Regression for classifying food choices based on nutritional information.
Developed by: POOJASRI.L
RegisterNumber: 212223220076 

# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset from the given URL
dataset_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML241EN-SkillsNetwork/labs/datasets/food_items.csv"
food_df = pd.read_csv(dataset_url)

# Check the column names to identify the target column
print("Column Names in the Dataset:")
print(food_df.columns)

# Display first few rows to inspect the dataset structure
print(food_df.head())

# Now, check the column names and find the correct target column.
# The target column should be something like 'food_class', 'is_diabetic_friendly', or a similar name.
# For example, let's assume we need to classify food based on a column named 'Diabetic_Friendly'
# Replace 'Diabetic_Friendly' with the actual column name after inspecting the dataset

# Find the correct target column after inspecting the dataset
target_column = 'class'  # Replace with actual column name after inspecting

# Ensure the target column exists
if target_column not in food_df.columns:
    raise KeyError(f"'{target_column}' not found in the dataset columns")

# Features and target variables
X = food_df.drop(target_column, axis=1)  # Features (excluding the target column)
y = food_df[target_column]  # Target variable (whether food is recommended or not)

# Handle missing values if there are any
X = X.fillna(X.mean())  # Replace NaN values with column mean (you can choose other strategies)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define logistic regression model with elastic-net penalty
# Set solver to 'saga' and specify an l1_ratio between 0 and 1 for elastic-net
elastic_net_model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000)

# Train the model
elastic_net_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = elastic_net_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Diabetic Friendly', 'Diabetic Friendly'], yticklabels=['Not Diabetic Friendly', 'Diabetic Friendly'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Classification Report
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

```

## Output:
![image](https://github.com/user-attachments/assets/55b5d46a-2551-4429-8e81-ada3acb341ed)

![image](https://github.com/user-attachments/assets/89cbe459-6745-4ed4-96d7-8ea9d61d0c78)

![image](https://github.com/user-attachments/assets/e9452a0e-9420-469c-911a-b42dc7d70e25)


## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
