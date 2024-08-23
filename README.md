# Setup

## GCP 
1. Create a GCP account with your work email
2. go to your terminal in your IDE
    * gcloud auth login
    * gcloud config set project union-ai-poc
        * only need to run once
    * gcloud auth application-default login
    * gcloud auth configure-docker us-east4-docker.pkg.dev

## Running Wine Classification Hyperparameter Workflow

* Local Run
    * union run wine_classification_hyperparameter_search.py training_workflow
* Run in Cloud
    * union run --remote wine_classification_hyperparameter_search.py training_workflow