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
import mlflow
from sklearn.utils.class_weight import compute_class_weight # Import compute_class_weight

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

# Dummy token for local testing if not in Colab/GH Actions
token = os.getenv("HF_TOKEN", "hf_dummy_token_for_testing")

# Login to Hugging Face Hub if a valid token is available
if token and token != "hf_dummy_token_for_testing":
    login(token=token, add_to_git_credential=False)

api = HfApi(token=token)

# Define the Hugging Face repository ID
HF_REPO_ID = "divyavarmamisc/tourism-prediction"

# Download train/test data from Hugging Face
Xtrain = pd.read_csv(hf_hub_download(repo_id=HF_REPO_ID, filename="Xtrain.csv", repo_type="dataset", token=token))
Xtest = pd.read_csv(hf_hub_download(repo_id=HF_REPO_ID, filename="Xtest.csv", repo_type="dataset", token=token))
ytrain = pd.read_csv(hf_hub_download(repo_id=HF_REPO_ID, filename="ytrain.csv", repo_type="dataset", token=token)).squeeze("columns")
ytest = pd.read_csv(hf_hub_download(repo_id=HF_REPO_ID, filename="ytest.csv", repo_type="dataset", token=token)).squeeze("columns")

# Define numeric and categorical features (must match prep.py)
numeric_features = [
    'Age',
    'DurationOfPitch',
    'NumberOfPersonVisiting',
    'NumberOfFollowups',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'NumberOfChildrenVisiting',
    'MonthlyIncome'
]
categorical_features = [
    'TypeofContact',
    'CityTier',
    'Occupation',
    'Gender',
    'ProductPitched',
    'MaritalStatus',
    'Passport',
    'PitchSatisfactionScore',
    'OwnCar',
    'Designation'
]

# Calculate class weights for binary imbalance
classes = ytrain.unique()
# Ensure classes are sorted for consistent compute_class_weight behavior
classes.sort()
class_weights_array = compute_class_weight(class_weight='balanced', classes=classes, y=ytrain)
class_weights_dict = {c: w for c, w in zip(classes, class_weights_array)}

# Create sample weights for the training data
sample_weights_train = ytrain.map(class_weights_dict)

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define base XGBoost model for binary classification
xgb_model = xgb.XGBClassifier(random_state=42, objective='binary:logistic')

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
    # Pass sample_weight to the fit method of GridSearchCV via fit_params
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

    # Make predictions
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    # Generate classification reports
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log the metrics for the best model (binary classification)
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision_1": train_report['1']['precision'], # Assuming '1' is the positive class
        "train_recall_1": train_report['1']['recall'],
        "train_f1-score_1": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision_1": test_report['1']['precision'],
        "test_recall_1": test_report['1']['recall'],
        "test_f1-score_1": test_report['1']['f1-score']
    })

    # Save the model locally
    model_path = "best_prod_taken_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "divyavarmamisc/prod_taken_model"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")
    except HfHubHTTPError as e:
        print(f"An error occurred during repo info or creation: {e}")
        if "401 Unauthorized" in str(e):
            print("Please check your Hugging Face token. It might be invalid or missing.")
        raise # Re-raise the exception after printing details

    api.upload_file(
        path_or_fileobj="best_prod_taken_model_v1.joblib",
        path_in_repo="best_prod_taken_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )

