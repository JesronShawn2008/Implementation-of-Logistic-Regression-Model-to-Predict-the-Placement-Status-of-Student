# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset into a DataFrame and separate the input features (CGPA, Internship, Communication_Score) and the target label (Placed).

2. Split the data into training and testing sets, then apply standard scaling to normalize the feature values.

3. Train a Logistic Regression model using the scaled training dataset.

4. Predict placement status and probabilities for the test data, and display the actual vs. predicted results.
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
data = {
    "CGPA": [8.5, 7.2, 6.8, 9.1, 5.9, 7.8, 8.0, 6.0, 9.3, 7.0],
    "Internship": [1, 0, 1, 1, 0, 1, 0, 0, 1, 0],   # 1 = Yes, 0 = No
    "Communication_Score": [8, 6, 7, 9, 5, 7, 8, 6, 9, 6],
    "Placed": [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]        # Target
}
df = pd.DataFrame(data)

X = df.drop("Placed", axis=1)
y = df["Placed"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)

print("Actual Placement Status:", list(y_test))
print("Predicted Placement Status:", list(y_pred))
print("\nPrediction Probabilities for each student:\n", y_prob)

Developed by: Jesron Shawn C J
RegisterNumber:  25012933
*/
```

## Output:
<img width="1049" height="160" alt="image" src="https://github.com/user-attachments/assets/42335500-795d-4543-b94c-9336e3b1182c" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
