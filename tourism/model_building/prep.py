import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib


bank_dataset = pd.read_csv("tourism/data/tourism.csv")
print("Dataset loaded successfully.")



# Encoding the categorical 'Type' column
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

target_col = 'NumberOfPersonVisiting'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col] - 1 # Subtract 1 to make target classes 0-indexed for XGBoost Classifier

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42)


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


# Removed class_weight calculation as it's for binary classification (0/1 target)
# and 'NumberOfPersonVisiting' is not a binary target with 0 as a class.
# class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
# class_weight

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define base XGBoost model
# Removed scale_pos_weight as it's for binary classification
xgb_model = xgb.XGBClassifier(random_state=42)


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

with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

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

    # The classification_threshold is typically used for binary classification.
    # For multi-class, predict_proba returns probabilities for each class,
    # and argmax is usually used to get the predicted class.
    # If 'NumberOfPersonVisiting' is intended as a categorical target,
    # predict() will directly return the class labels.
    # Keeping threshold for consistency, but be aware of its multi-class interpretation.
    classification_threshold = 0.45 # This threshold might need adjustment for multi-class

    y_pred_train_proba = best_model.predict_proba(Xtrain)
    # Assuming 1 is a relevant class for binary-like evaluation within multi-class context
    # For true multi-class, y_pred_train should be best_model.predict(Xtrain)
    y_pred_train = (y_pred_train_proba[:, best_model.classes_.tolist().index(1)] >= classification_threshold).astype(int) if 1 in best_model.classes_ else best_model.predict(Xtrain)

    y_pred_test_proba = best_model.predict_proba(Xtest)
    y_pred_test = (y_pred_test_proba[:, best_model.classes_.tolist().index(1)] >= classification_threshold).astype(int) if 1 in best_model.classes_ else best_model.predict(Xtest)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log the metrics. Adjusting to handle cases where class '1' might not exist or interpretation differs.
    logged_metrics = {
        "train_accuracy": train_report['accuracy'],
        "test_accuracy": test_report['accuracy']
    }
    if '1' in train_report: # Check if class '1' exists in reports for precision/recall/f1
        logged_metrics.update({
            "train_precision": train_report['1']['precision'],
            "train_recall": train_report['1']['recall'],
            "train_f1-score": train_report['1']['f1-score']
        })
    if '1' in test_report:
        logged_metrics.update({
            "test_precision": test_report['1']['precision'],
            "test_recall": test_report['1']['recall'],
            "test_f1-score": test_report['1']['f1-score']
        })

    mlflow.log_metrics(logged_metrics)
