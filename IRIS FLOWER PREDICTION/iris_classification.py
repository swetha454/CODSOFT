import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("IRIS.csv")

print("First 5 Rows")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

# -----------------------------
# Data Visualization
# -----------------------------

sns.pairplot(df, hue='species')
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df.drop("species", axis=1).corr(), annot=True, cmap="Blues")
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------
# Prepare Data
# -----------------------------

X = df.drop("species", axis=1)
y = df["species"]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# -----------------------------
# Split Dataset
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

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------

y_pred = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report")
print(classification_report(y_test, y_pred,
                            target_names=encoder.classes_))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm,
            annot=True,
            cmap="Greens",
            fmt="d",
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# -----------------------------
# Save Model
# -----------------------------

joblib.dump(model, "model.pkl")

print("\nModel Saved Successfully!")

# -----------------------------
# Predict New Flower
# -----------------------------

print("\nPredict New Flower")

sl = float(input("Sepal Length: "))
sw = float(input("Sepal Width : "))
pl = float(input("Petal Length: "))
pw = float(input("Petal Width : "))

sample = [[sl, sw, pl, pw]]

prediction = model.predict(sample)

flower = encoder.inverse_transform(prediction)

print("\nPredicted Species:", flower[0])