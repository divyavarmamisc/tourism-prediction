# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from google.colab import userdata # Import userdata
import mlflow
from sklearn.utils.class_weight import compute_class_weight # Import compute_class_weight

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

# Retrieve and strip the token from Colab secrets
raw_token = userdata.get('HF_TOKEN')
token = raw_token.strip()

api = HfApi(token=token)

# Define the Hugging Face repository ID
HF_REPO_ID = "divyavarmamisc/tourism-prediction"

# Download and load the data files explicitly
try:
    Xtrain_local_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="Xtrain.csv",
        repo_type="dataset",
        token=token
    )
    Xtest_local_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="Xtest.csv",
        repo_type="dataset",
        token=token
    )
    ytrain_local_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="ytrain.csv",
        repo_type="dataset",
        token=token
    )
    ytest_local_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="ytest.csv",
        repo_type="dataset",
        token=token
    )

    Xtrain = pd.read_csv(Xtrain_local_path)
    Xtest = pd.read_csv(Xtest_local_path)
    # FIX: Load ytrain and ytest as Series from the 'NumberOfPersonVisiting' column
    ytrain = pd.read_csv(ytrain_local_path)['NumberOfPersonVisiting'].squeeze()
    ytest = pd.read_csv(ytest_local_path)['NumberOfPersonVisiting'].squeeze()
    print("All datasets loaded successfully from Hugging Face.")
except Exception as e:
    print(f"Error downloading or loading datasets: {e}")
    raise


# One-hot encode 'Type' and scale numeric features
numeric_features = [
    'NumberOfFollowups',
    'NumberOfTrips',
    'PreferredPropertyStar',
    'PitchSatisfactionScore',
    'MonthlyIncome',
    'NumberOfChildrenVisiting'
]
categorical_features = ['Gender']


# FIX: Remove inappropriate scale_pos_weight calculation for multi-class
# Calculate class weights for multi-class imbalance
classes = ytrain.unique()
class_weights_array = compute_class_weight(class_weight='balanced', classes=classes, y=ytrain)
class_weights_dict = {c: w for c, w in zip(classes, class_weights_array)}

# Create sample weights for the training data
sample_weights_train = ytrain.map(class_weights_dict)


# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define base XGBoost model
# FIX: Remove scale_pos_weight as it's for binary classification. Use objective for multi-class.
xgb_model = xgb.XGBClassifier(random_state=42, objective='multi:softmax')

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    # FIX: Pass sample_weight to the fit method of GridSearchCV via fit_params
    grid_search.fit(Xtrain, ytrain, xgbclassifier__sample_weight=sample_weights_train)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    # FIX: Simplify prediction for multi-class and remove binary threshold logic
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    # Use classification_report directly on multi-class predictions
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log the metrics for the best model
    # FIX: Adjust logged metrics for multi-class, using 'macro avg' for aggregated metrics.
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_macro_precision": train_report['macro avg']['precision'],
        "train_macro_recall": train_report['macro avg']['recall'],
        "train_macro_f1-score": train_report['macro avg']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_macro_precision": test_report['macro avg']['precision'],
        "test_macro_recall": test_report['macro avg']['recall'],
        "test_macro_f1-score": test_report['macro avg']['f1-score']
    })

    # Save the model locally
    model_path = "best_machine_failure_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "praneeth232/machine_failure_model"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_machine_failure_model_v1.joblib",
        path_in_repo="best_machine_failure_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
