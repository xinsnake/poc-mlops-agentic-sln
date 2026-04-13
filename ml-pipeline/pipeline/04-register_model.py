"""
register_model.py

Registers the trained model artifact from an AML job into the AML Model Registry.

After training completes, the model.txt file sits in the job's outputs/ folder.
This script promotes it to a named, versioned model asset — decoupled from the job —
so it can be referenced and deployed by name.

The latest completed training job is resolved automatically — no need to hardcode
the job name after each retraining run.
"""

import os
from dotenv import load_dotenv
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

load_dotenv()

TRAINING_JOB_DISPLAY_NAME = "anomaly-classifier-training"  # set in 03-submit_training.py
MODEL_NAME = os.environ.get("AML_MODEL_NAME", "anomaly-classifier")


def get_latest_completed_job(ml_client: MLClient) -> str:
    """
    Find the most recently completed training job by display name.
    AML assigns random names (e.g. happy_onion_...) to each run — this resolves
    the latest one automatically so the script works without manual updates.
    """
    jobs = ml_client.jobs.list()
    completed = [
        j for j in jobs
        if getattr(j, "display_name", None) == TRAINING_JOB_DISPLAY_NAME
        and j.status == "Completed"
    ]
    if not completed:
        raise RuntimeError(
            f"No completed jobs found with display name '{TRAINING_JOB_DISPLAY_NAME}'. "
            "Run 03-submit_training.py first."
        )
    # Sort explicitly by creation time descending — don't rely on API return order
    latest = sorted(completed, key=lambda j: j.creation_context.created_at, reverse=True)[0]
    print(f"Latest completed job: {latest.name}")
    return latest.name


def register_model(ml_client: MLClient) -> None:
    training_job_name = get_latest_completed_job(ml_client)

    # Get the job to confirm it completed successfully
    job = ml_client.jobs.get(training_job_name)
    print(f"Job:    {job.name}")
    print(f"Status: {job.status}")

    # The model artifact path — AML job outputs are addressable via azureml: URI
    model_path = f"azureml://jobs/{training_job_name}/outputs/artifacts/paths/outputs/model.txt"

    model = Model(
        path=model_path,
        name=MODEL_NAME,
        type=AssetTypes.CUSTOM_MODEL,
        description=(
            "LightGBM binary classifier for forecast anomaly classification (UC4). "
            "Classifies anomalies as 'temporary' (0) or 'baseline_shift' (1). "
            "Trained on synthetic Snowflake data — spike PoC."
        ),
        tags={
            "framework": "lightgbm",
            "task": "binary-classification",
            "target": "anomaly_type",
            "training_job": training_job_name,
            "spike": "true",
        },
    )

    registered = ml_client.models.create_or_update(model)
    print(f"\nModel registered successfully!")
    print(f"  Name:    {registered.name}")
    print(f"  Version: {registered.version}")
    print(f"  ID:      {registered.id}")
    print(f"\nView in AML Studio:")
    print(f"  https://ml.azure.com/models/{registered.name}/version/{registered.version}"
          f"?wsid=/subscriptions/{os.environ['AZURE_SUBSCRIPTION_ID']}"
          f"/resourcegroups/{os.environ['AZURE_RESOURCE_GROUP']}"
          f"/workspaces/{os.environ['AZURE_ML_WORKSPACE']}")


if __name__ == "__main__":
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_ML_WORKSPACE"],
    )

    register_model(ml_client)
