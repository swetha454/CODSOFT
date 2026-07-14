# Movie Rating Prediction using Machine Learning
# CodSoft Data Science Internship

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("IMDb Movies India.csv", encoding="latin1")

print("First 5 Rows")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nColumns")
print(df.columns)

# -----------------------------
# Check Missing Values
# -----------------------------
print("\nMissing Values")
print(df.isnull().sum())

# -----------------------------
# Data Cleaning
# -----------------------------

# Remove rows where Rating is missing
df = df.dropna(subset=["Rating"])

# Fill missing values in categorical columns
categorical_columns = [
    "Genre",
    "Director",
    "Actor 1",
    "Actor 2",
    "Actor 3"
]

for col in categorical_columns:
    df[col] = df[col].fillna("Unknown")

# Fill missing values in numeric columns
df["Votes"] = df["Votes"].fillna("0")
df["Duration"] = df["Duration"].fillna("0 min")

# Remove commas from Votes
df["Votes"] = df["Votes"].astype(str).str.replace(",", "")

# Keep only numbers
df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce")

# Remove " min" from Duration
df["Duration"] = df["Duration"].astype(str).str.replace(" min", "")

df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce")

# Fill remaining numeric missing values
df["Votes"] = df["Votes"].fillna(df["Votes"].median())
df["Duration"] = df["Duration"].fillna(df["Duration"].median())

# -----------------------------
# Exploratory Data Analysis
# -----------------------------

plt.figure(figsize=(8,5))
sns.histplot(df["Rating"], bins=20, color="skyblue")
plt.title("Distribution of Movie Ratings")
plt.show()

plt.figure(figsize=(10,6))
df["Genre"].value_counts().head(10).plot(kind="bar")
plt.title("Top 10 Genres")
plt.show()

plt.figure(figsize=(10,6))
df["Director"].value_counts().head(10).plot(kind="bar")
plt.title("Top 10 Directors")
plt.show()

# -----------------------------
# Label Encoding
# -----------------------------

encoder = LabelEncoder()

df["Genre"] = encoder.fit_transform(df["Genre"])
df["Director"] = encoder.fit_transform(df["Director"])
df["Actor 1"] = encoder.fit_transform(df["Actor 1"])
df["Actor 2"] = encoder.fit_transform(df["Actor 2"])
df["Actor 3"] = encoder.fit_transform(df["Actor 3"])

# -----------------------------
# Select Features
# -----------------------------

X = df[
    [
        "Genre",
        "Director",
        "Actor 1",
        "Actor 2",
        "Actor 3",
        "Duration",
        "Votes"
    ]
]

y = df["Rating"]

# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Train Model
# -----------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------

prediction = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------

mae = mean_absolute_error(y_test, prediction)
rmse = np.sqrt(mean_squared_error(y_test, prediction))
r2 = r2_score(y_test, prediction)

print("\nModel Performance")
print("---------------------------")
print("Mean Absolute Error :", mae)
print("Root Mean Square Error :", rmse)
print("R2 Score :", r2)

# -----------------------------
# Actual vs Predicted
# -----------------------------

result = pd.DataFrame({
    "Actual Rating": y_test,
    "Predicted Rating": prediction
})

print("\nSample Predictions")
print(result.head(10))

# -----------------------------
# Feature Importance
# -----------------------------

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

importance.sort_values().plot(
    kind="barh",
    figsize=(8,5)
)

plt.title("Feature Importance")
plt.show()