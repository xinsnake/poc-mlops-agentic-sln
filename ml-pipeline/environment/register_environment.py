"""
register_environment.py

Registers a custom AML Environment from conda.yml.
Run once (or whenever conda.yml changes) to create a new version.

Usage:
    cd ml-pipeline
    python environment/register_environment.py
"""

import os
from dotenv import load_dotenv
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

load_dotenv()

ENV_NAME = "anomaly-classifier-env"


def register():
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_ML_WORKSPACE"],
    )

    conda_file = os.path.join(os.path.dirname(__file__), "conda.yml")

    env = Environment(
        name=ENV_NAME,
        description="Custom environment for anomaly classifier training (LightGBM + MLflow)",
        conda_file=conda_file,
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    )

    registered = ml_client.environments.create_or_update(env)
    print(f"Environment registered: {registered.name}:{registered.version}")
    print(f"\nUpdate your .env:")
    print(f"  AML_ENVIRONMENT=azureml:{registered.name}:{registered.version}")


if __name__ == "__main__":
    register()
