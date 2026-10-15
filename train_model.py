
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Find the dataset beside this Python file
folder = Path(__file__).resolve().parent
data_path = folder / "Churn_Modelling.csv"

if not data_path.exists():
    raise FileNotFoundError(
        "Churn_Modelling.csv is missing. "
        "Copy the dataset into the project folder."
    )

# 1. Load and prepare the dataset
df = pd.read_csv(data_path)

df.drop(
    columns=["RowNumber", "CustomerId", "Surname"],
    errors="ignore",
    inplace=True
)

df = pd.get_dummies(
    df,
    columns=["Geography", "Gender"],
    drop_first=True
)

# 2. Separate features and target
X = df.drop(columns=["Exited"])
y = df["Exited"]

feature_columns = X.columns.tolist()

# 3. Split training data and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 4. Fit the scaler on training data only
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# 6. Evaluate the model on unseen test data
predictions = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, predictions)
print(f"Test accuracy: {accuracy:.2%}")

# 7. Save all artifacts together
joblib.dump(model, folder / "random_forest_model.pkl")
joblib.dump(scaler, folder / "scaler.pkl")
joblib.dump(feature_columns, folder / "feature_columns.pkl")

print("Model, scaler and feature columns saved successfully.")
