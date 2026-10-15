# Customer Churn Intelligence — Streamlit app

A polished Streamlit interface for your Bank Customer Churn Prediction Random Forest model.

## Folder structure

Place these files in one project folder:

```text
customer_churn_streamlit/
├── app.py
├── requirements.txt
├── README.md
├── NOTEBOOK_CHANGES.md
├── random_forest_model.pkl    # create/download from your updated Colab notebook
├── scaler.pkl                 # create/download from your updated Colab notebook
├── feature_columns.pkl        # create/download from your updated Colab notebook
└── .streamlit/
    └── config.toml
```

The `.pkl` files are not included in this starter package because the trained model files were not part of the uploaded notebook file. The app intentionally gives a helpful setup message until you add your own model artifacts.

## Step 1 — Update the notebook and create the model files

1. Open `Customer_Churn_Prediction.ipynb` in Google Colab.
2. Follow all four steps in `NOTEBOOK_CHANGES.md`. The important change is to remove `RowNumber` before training, then save `feature_columns` with the model and scaler.
3. Run **Runtime → Run all**. Check that training and evaluation finish successfully.
4. In Colab's **Files** panel, download these three files: `random_forest_model.pkl`, `scaler.pkl`, and `feature_columns.pkl`.
5. Copy all three downloaded files into the same folder as `app.py`. Do not rename them.

The model, scaler and feature list must come from the same training run. Do not mix old and new `.pkl` files.

## Step 2 — Check the scikit-learn version

The version used to save a scikit-learn model should match the version used to load it. In a notebook cell, run:

```python
import sklearn
print(sklearn.__version__)
```

This starter `requirements.txt` currently pins `scikit-learn==1.6.1`. If Colab prints a different version, replace `1.6.1` in `requirements.txt` with the exact version printed by Colab, then save the file. If you later retrain the model with another scikit-learn version, update the requirement to match that version too.

## Step 3 — Test on your computer (Windows + VS Code)

1. Install Python **3.12** if you do not already have it. It avoids compatibility issues that may occur if the only local Python installation is very new.
2. Extract the ZIP and open the extracted `customer_churn_streamlit` folder in VS Code using **File → Open Folder**.
3. Open **Terminal → New Terminal**.
4. Create and activate a virtual environment:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt in VS Code and run:

   ```bat
   .venv\Scripts\activate.bat
   ```

5. Install the packages:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

6. Start the app:

   ```bash
   python -m streamlit run app.py
   ```

7. A browser tab should open at a local address like `http://localhost:8501`. Test the form and prediction. This local address is only on your computer; it is not the link to send to your teacher.

If the app reports missing `.pkl` files, make sure the three files are in the same folder as `app.py`. If it reports a feature mismatch, re-run the notebook edits and download all three artifacts from that same run.

## Step 4 — Upload the project to GitHub

1. Sign in to GitHub and create a new repository, for example `customer-churn-intelligence`.
2. For the simplest public sharing, set the repository to **Public**. Do not upload secrets, API keys, or private customer information. The original CSV is not required to run the deployed prediction app.
3. Upload `app.py`, `requirements.txt`, the `.streamlit/config.toml` file, `README.md`, `NOTEBOOK_CHANGES.md`, and the three `.pkl` files. Keep the `.streamlit` folder name and structure exactly as shown.
4. Confirm that `app.py` is at the repository root, not inside an extra nested folder.

You can upload through GitHub's website using **Add file → Upload files**, or use Git from VS Code if you already know it.

## Step 5 — Deploy to Streamlit Community Cloud

1. Open [Streamlit Community Cloud](https://share.streamlit.io/).
2. Sign in and connect your GitHub account if asked.
3. Click **Create app** and choose the option for an existing app/repository.
4. Select your GitHub repository, branch `main` (or whichever branch you uploaded), and main file path `app.py`.
5. In **Advanced settings**, select Python 3.12. Make sure `requirements.txt` has the exact scikit-learn version that you checked in Step 2.
6. If the custom app URL field is available, enter a short unique name such as `customer-churn-intelligence-yourinitials`.
7. Click **Deploy** and wait for the build to complete. The first deployment may take a few minutes.
8. Open the `https://...streamlit.app` URL, test a prediction, and send that live URL to your teacher.

Official deployment guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

## Step 6 — After deployment

- The public URL is the link to share; `localhost:8501` is not.
- When you edit and push files to GitHub, Streamlit Community Cloud usually redeploys the app automatically.
- If deployment fails, open the app's deployment logs in Community Cloud. Read the first error near the bottom; common causes are a missing `.pkl` file, an incorrect entrypoint path, or a scikit-learn version mismatch.
- Treat the predicted probability as a model estimate, not a guarantee. The app is an academic demonstration, not a production banking decision system.

## Current model assumptions

This app matches the supplied notebook's preprocessing: `pd.get_dummies(..., drop_first=True)` and `StandardScaler` followed by a `RandomForestClassifier(n_estimators=100, random_state=42)`. The expected encoded inputs are `Geography_Germany`, `Geography_Spain`, and `Gender_Male`; France and Female are baseline categories. If you change preprocessing in the notebook, update the app so its inputs and feature names remain consistent.
