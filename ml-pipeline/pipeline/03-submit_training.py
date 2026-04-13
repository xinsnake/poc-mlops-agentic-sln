"""
submit_training.py

Submits train.py as an AML Command Job.

This runs locally — it tells AML to:
1. Spin up a compute cluster
2. Install dependencies
3. Run train.py with the registered Data Assets as inputs
4. Log metrics to AML Experiments
5. Save the trained model to AML outputs

After this completes, the trained model appears in the AML Job outputs,
ready to be registered in the Model Registry (done in deploy_endpoint.py).
"""

import os
from dotenv import load_dotenv
from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.entities import AmlCompute
from azure.identity import DefaultAzureCredential

load_dotenv()

COMPUTE_NAME = os.environ.get("AML_COMPUTE_NAME", "spike-cluster")
COMPUTE_SIZE = os.environ.get("AML_COMPUTE_SIZE", "Standard_DS2_v2")


def get_or_create_compute(ml_client: MLClient) -> str:
    """Get existing compute cluster or create a new one."""
    try:
        ml_client.compute.get(COMPUTE_NAME)
        print(f"Using existing compute cluster: {COMPUTE_NAME}")
    except Exception:
        print(f"Creating compute cluster: {COMPUTE_NAME} ({COMPUTE_SIZE})")
        compute = AmlCompute(
            name=COMPUTE_NAME,
            size=COMPUTE_SIZE,
            min_instances=0,   # scales to zero when idle (no cost)
            max_instances=1,
        )
        ml_client.compute.begin_create_or_update(compute).result()
        print(f"Compute cluster created: {COMPUTE_NAME}")
    return COMPUTE_NAME


def submit_job(ml_client: MLClient, compute_name: str) -> None:
    """Submit the training job to AML."""

    # Resolve latest data asset versions dynamically — avoids hardcoding
    # which breaks every time 02-data_pipeline.py runs and creates a new version
    train_version = ml_client.data.get(name="anomaly-train", label="latest").version
    val_version = ml_client.data.get(name="anomaly-validation", label="latest").version
    print(f"Using data assets: anomaly-train:{train_version}, anomaly-validation:{val_version}")

    train_data = Input(
        type=AssetTypes.URI_FILE,
        path=f"azureml:anomaly-train:{train_version}",
        mode=InputOutputModes.DOWNLOAD,
    )
    validation_data = Input(
        type=AssetTypes.URI_FILE,
        path=f"azureml:anomaly-validation:{val_version}",
        mode=InputOutputModes.DOWNLOAD,
    )

    job = command(
        display_name="anomaly-classifier-training",
        description="LightGBM training job for forecast anomaly classification (UC4 spike)",
        code="./src",                      # uploads only the src/ folder to AML
        command=(
            "python train.py"
            " --train-data ${{inputs.train_data}}"
            " --validation-data ${{inputs.validation_data}}"
            " --n-estimators 200"
            " --learning-rate 0.05"
            " --num-leaves 31"
        ),
        inputs={
            "train_data": train_data,
            "validation_data": validation_data,
        },
        environment=os.environ["AML_ENVIRONMENT"],
        compute=compute_name,
    )

    submitted = ml_client.jobs.create_or_update(job)
    print(f"\nJob submitted: {submitted.display_name}")
    print(f"Job name:      {submitted.name}")
    print(f"Status:        {submitted.status}")
    print(f"\nView in AML Studio:")
    print(f"  https://ml.azure.com/runs/{submitted.name}?wsid=/subscriptions/{os.environ['AZURE_SUBSCRIPTION_ID']}/resourcegroups/{os.environ['AZURE_RESOURCE_GROUP']}/workspaces/{os.environ['AZURE_ML_WORKSPACE']}")
    print(f"\nStreaming logs (Ctrl+C to stop following, job continues running)...")

    # Stream logs so you can watch progress in the terminal
    ml_client.jobs.stream(submitted.name)


if __name__ == "__main__":
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_ML_WORKSPACE"],
    )

    compute_name = get_or_create_compute(ml_client)
    submit_job(ml_client, compute_name)
