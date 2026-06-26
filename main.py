# import pandas as pd

# # Simple dataset
# data = {
#     "StudyTime": [2, 4, 6, 8],
#     "Absences": [10, 5, 2, 1],
#     "FinalScore": [50, 60, 75, 90]
# }

# df = pd.DataFrame(data)

# print("Dataset:")
# print(df)





#TRAINING
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression

# # Dataset
# data = {
#     "StudyTime": [2, 4, 6, 8, 1, 3, 5, 7],
#     "Absences": [10, 5, 2, 1, 12, 7, 3, 2],
#     "FinalScore": [50, 60, 75, 90, 45, 55, 70, 85]
# }

# df = pd.DataFrame(data)

# # Features (inputs) and Target (output)
# X = df[["StudyTime", "Absences"]]
# y = df["FinalScore"]

# # Split data (training + testing)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # Create model
# model = LinearRegression()

# # Train model
# model.fit(X_train, y_train)

# # Make predictions
# predictions = model.predict(X_test)

# print("Predictions:", predictions)
# print("Actual:", list(y_test))





#idk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# -----------------------------
# STEP 1: Create dataset
# -----------------------------
data = {
    "StudyTime": [1,2,3,4,5,6,7,8,2,3,4,5,6,7,8,1],
    "Absences": [12,10,8,6,5,3,2,1,11,9,7,5,3,2,1,13],
    "FinalScore": [40,50,55,60,65,75,85,90,45,52,58,63,78,88,92,35]
}

df = pd.DataFrame(data)

# -----------------------------
# STEP 2: Define inputs & output
# -----------------------------
X = df[["StudyTime", "Absences"]]   # features (input)
y = df["FinalScore"]                # target (output)

# -----------------------------
# STEP 3: Split data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# STEP 4: Linear Regression model
# -----------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

# -----------------------------
# STEP 5: Decision Tree model
# -----------------------------
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

# -----------------------------
# STEP 6: Evaluate models
# -----------------------------
lr_score = r2_score(y_test, lr_pred)
dt_score = r2_score(y_test, dt_pred)

print("Linear Regression Predictions:", lr_pred)
print("Decision Tree Predictions:", dt_pred)

print("Actual values:", list(y_test))

print("Linear Regression R2 Score:", lr_score)
print("Decision Tree R2 Score:", dt_score)


import matplotlib.pyplot as plt

# Plot actual vs predicted
plt.scatter(y_test, lr_pred)
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")
plt.title("Linear Regression: Actual vs Predicted")

plt.show()