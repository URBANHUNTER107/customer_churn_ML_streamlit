# Required notebook edits before running the app

Your notebook currently drops `CustomerId` and `Surname`, but it leaves `RowNumber` in the feature set. The deployed app should not ask a user for an arbitrary row number, so retrain the model without it and save the feature names used during training.

## 1. Update your preprocessing cell

Replace the preprocessing cell with:

```python
df.drop(columns=["RowNumber", "CustomerId", "Surname"], errors="ignore", inplace=True)
df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)
```

This safely removes the dataset's row index along with identifiers that should not be used as customer predictors.

## 2. Update the cell where you define `X` and `y`

Use:

```python
X = df.drop("Exited", axis=1)
y = df["Exited"]
feature_columns = X.columns.tolist()
```

Keep your existing 80/20 train-test split, fit `StandardScaler` on `X_train` only, and train the Random Forest on the scaled training data as you do now.

## 3. Update your model-saving cell

Use:

```python
joblib.dump(model, "random_forest_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_columns, "feature_columns.pkl")
print("Saved model, scaler, and feature columns successfully.")
```

## 4. Re-run from the start

In Colab, use **Runtime → Run all**. Confirm the model trains and the evaluation cell runs successfully. The three `.pkl` files should then appear in the Files panel.

You can optionally improve the evaluation cell to include precision, recall, F1-score and the confusion matrix:

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
```

Don't report an accuracy number in the website unless you have verified it from your test-set results.
