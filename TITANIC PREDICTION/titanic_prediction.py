# ============================================
# TITANIC SURVIVAL PREDICTION
# CodSoft Data Science Internship
# ============================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --------------------------------------------
# Load Dataset
# --------------------------------------------

df = pd.read_csv("Titanic-Dataset.csv")

print("========== First 5 Rows ==========")
print(df.head())

print("\n========== Dataset Shape ==========")
print(df.shape)

print("\n========== Dataset Information ==========")
print(df.info())

print("\n========== Missing Values ==========")
print(df.isnull().sum())

# --------------------------------------------
# Data Cleaning
# --------------------------------------------

# Fill missing Age values with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked values with mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Fill missing Fare values with median
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# Drop Cabin column because most values are missing
df.drop("Cabin", axis=1, inplace=True)

print("\n========== Missing Values After Cleaning ==========")
print(df.isnull().sum())

# --------------------------------------------
# Exploratory Data Analysis
# --------------------------------------------

# Survival Count
plt.figure(figsize=(6,4))
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.show()

# Gender vs Survival
plt.figure(figsize=(6,4))
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Gender vs Survival")
plt.show()

# Passenger Class vs Survival
plt.figure(figsize=(6,4))
sns.countplot(x="Pclass", hue="Survived", data=df)
plt.title("Passenger Class vs Survival")
plt.show()

# Age Distribution
plt.figure(figsize=(7,4))
sns.histplot(df["Age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# Correlation Heatmap
numeric_df = df.select_dtypes(include=["number"])

plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="Blues")
plt.title("Correlation Heatmap")
plt.show()

# --------------------------------------------
# Convert Categorical Data
# --------------------------------------------

encoder = LabelEncoder()

df["Sex"] = encoder.fit_transform(df["Sex"])
df["Embarked"] = encoder.fit_transform(df["Embarked"])

# --------------------------------------------
# Select Features
# --------------------------------------------

X = df[[
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]]

y = df["Survived"]

# --------------------------------------------
# Split Dataset
# --------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------------------------
# Train Model
# --------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# --------------------------------------------
# Prediction
# --------------------------------------------

y_pred = model.predict(X_test)

# --------------------------------------------
# Model Evaluation
# --------------------------------------------

print("\n========== MODEL PERFORMANCE ==========")

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy :", round(accuracy * 100, 2), "%")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# --------------------------------------------
# Actual vs Predicted
# --------------------------------------------

result = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

print("\n========== Sample Predictions ==========")
print(result.head(10))

# --------------------------------------------
# Feature Importance
# --------------------------------------------

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

importance = importance.sort_values()

plt.figure(figsize=(8,5))
importance.plot(kind="barh")
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.show()

print("\nProject Completed Successfully!")