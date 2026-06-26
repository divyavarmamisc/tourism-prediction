from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
from google.colab import userdata
import os


repo_id = "divyavarmamisc/tourism-prediction"
repo_type = "dataset"

# Initialize API client with token from Colab secrets
api = HfApi(token=userdata.get("HF_TOKEN"))

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

api.upload_folder(
    folder_path="tourism/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
