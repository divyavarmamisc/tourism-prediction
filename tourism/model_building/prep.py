import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from huggingface_hub import login, HfApi, hf_hub_download
# for model training, tuning, and evaluation
import xgboost as xgb

from sklearn.model_selection import GridSearchCV
import mlflow
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
from sklearn.utils.class_weight import compute_class_weight # Import compute_class_weight

# For Hugging Face authentication and API access
# Ensure HF_TOKEN is available as an environment variable or via userdata in Colab
# For script execution in environments like GitHub Actions, an env var is preferred.
# For local execution in Colab, userdata.get('HF_TOKEN') would be used.
# Here we assume HF_TOKEN is set in the environment or passed when executing the script.

# Dummy token for local testing if not in Colab/GH Actions
token = os.getenv("HF_TOKEN", "hf_dummy_token_for_testing")

# Login to Hugging Face Hub if a valid token is available
if token and token != "hf_dummy_token_for_testing":
    login(token=token, add_to_git_credential=False)

api = HfApi(token=token)

HF_REPO_ID = "divyavarmamisc/tourism-prediction"
HF_DATASET_FILENAME = "tourism.csv"

# Download the dataset file explicitly using hf_hub_download
try:
    dataset_local_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_DATASET_FILENAME,
        repo_type="dataset",
        token=token # Explicitly pass the token
    )
    df = pd.read_csv(dataset_local_path)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error downloading or loading dataset: {e}")
    raise

# Define target and features
target_col = 'ProdTaken'
id_cols = ['Unnamed: 0', 'CustomerID'] # Columns to be dropped as they are identifiers

# Separate target from features
y = df[target_col]
# Remove target and identifier columns from features
X = df.drop(columns=[target_col] + id_cols)

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save processed data to CSV files
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Upload processed data to Hugging Face Hub
files_to_upload = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files_to_upload:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id=HF_REPO_ID,
        repo_type="dataset",
    )
