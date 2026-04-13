"""
deploy_endpoint.py

Deploys the registered anomaly-classifier model to an AML Managed Online Endpoint.

What this creates:
  - An endpoint: a stable HTTPS URL that never changes
  - A deployment (blue): the actual VM + model + scoring script behind that URL

After this runs, you can POST a JSON payload to the endpoint URL
and receive a prediction (temporary / baseline_shift) with a probability score.

This is the component that UC4 (Forecast Parameter Agent) would call in production.
"""

import os
from dotenv import load_dotenv
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    CodeConfiguration,
)
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

load_dotenv()

ENDPOINT_NAME = os.environ.get("AML_ENDPOINT_NAME", "anomaly-classifier-endpoint")
DEPLOYMENT_NAME = "blue"        # standard convention — blue/green for safe rollouts
MODEL_NAME = os.environ.get("AML_MODEL_NAME", "anomaly-classifier")


def upload_and_register_score_script(ml_client: MLClient) -> str:
    """
    Upload score.py to blob storage using DefaultAzureCredential (resolves to az login user identity).
    The storage account blocks SAS-based uploads (allowSharedKeyAccess: false),
    so we upload directly via BlobServiceClient, then register as a Code asset.

    Returns the azureml:name:version reference for use in CodeConfiguration.
    """
    from azure.ai.ml.entities._assets._artifacts.code import Code

    datastore = ml_client.datastores.get_default()
    blob_base_url = f"https://{datastore.account_name}.blob.core.windows.net"
    blob_service = BlobServiceClient(account_url=blob_base_url, credential=DefaultAzureCredential())
    container_client = blob_service.get_container_client(datastore.container_name)

    # Upload score.py (path is relative to this script file)
    score_path = os.path.join(os.path.dirname(__file__), "..", "src", "score.py")

    # Register the blob folder as a Code asset (must use https:// URI, not azureml://)
    # Probe for the current highest version by trying get() calls, then increment.
    # Each version uses a unique blob path so AML always snapshots fresh content.
    next_version = "1"
    for candidate in range(50, 0, -1):
        try:
            ml_client._code.get(name="spike-score-code", version=str(candidate))
            next_version = str(candidate + 1)
            break
        except Exception:
            continue

    blob_name = f"spike-code/v{next_version}/score.py"
    with open(score_path, "rb") as f:
        container_client.upload_blob(name=blob_name, data=f, overwrite=True)
    print(f"  Uploaded score.py → blob: {blob_name}")

    blob_uri = f"{blob_base_url}/{datastore.container_name}/spike-code/v{next_version}/"
    code = Code(name="spike-score-code", version=next_version, path=blob_uri)
    result = ml_client._code.create_or_update(code)
    print(f"  Registered Code asset: {result.name}:{result.version}")

    return f"azureml:{result.name}:{result.version}"


def deploy(ml_client: MLClient, credential: DefaultAzureCredential) -> None:

    # ── Step 1: Create the endpoint (the stable HTTPS URL) ──────────────────
    print("Step 1: Creating endpoint...")
    endpoint = ManagedOnlineEndpoint(
        name=ENDPOINT_NAME,
        description="Forecast anomaly classifier — UC4 spike PoC",
        auth_mode="key",        # API key auth (simple for PoC)
    )
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"  Endpoint created: {ENDPOINT_NAME}")

    # ── Step 1b: Upload + register score.py (bypasses SAS restriction) ───────
    print("\nStep 1b: Uploading and registering score.py...")
    code_ref = upload_and_register_score_script(ml_client)
    print(f"  Code reference: {code_ref}")

    # ── Step 2: Create the deployment (VM + model + scoring script) ──────────
    # Resolve latest model version dynamically — avoids hardcoding which breaks
    # every time 04-register_model.py runs and creates a new version
    model_version = ml_client.models.get(name=MODEL_NAME, label="latest").version
    print(f"\nStep 2: Creating deployment with {MODEL_NAME}:{model_version} (this takes ~10 minutes)...")
    deployment = ManagedOnlineDeployment(
        name=DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME,
        model=f"azureml:{MODEL_NAME}:{model_version}",
        code_configuration=CodeConfiguration(
            code=code_ref,
            scoring_script="score.py",
        ),
        environment=os.environ["AML_ENVIRONMENT"],
        instance_type=os.environ.get("AML_COMPUTE_SIZE", "Standard_DS2_v2"),
        instance_count=1,
    )
    ml_client.online_deployments.begin_create_or_update(deployment).result()
    print(f"  Deployment created: {DEPLOYMENT_NAME}")

    # ── Step 3: Route 100% of traffic to the blue deployment ─────────────────
    print("\nStep 3: Routing traffic to deployment...")
    endpoint.traffic = {DEPLOYMENT_NAME: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print("  Traffic routed: 100% → blue")

    # ── Step 4: Print endpoint details ───────────────────────────────────────
    endpoint = ml_client.online_endpoints.get(ENDPOINT_NAME)
    keys = ml_client.online_endpoints.get_keys(ENDPOINT_NAME)

    print(f"\n{'='*60}")
    print(f"Endpoint deployed successfully!")
    print(f"  URL:     {endpoint.scoring_uri}")
    print(f"  API Key: {keys.primary_key}")
    print(f"\nTest with:")
    print(f"""  curl -X POST "{endpoint.scoring_uri}" \\
    -H "Authorization: Bearer {keys.primary_key}" \\
    -H "Content-Type: application/json" \\
    -d '{{
      "forecast_bias": 0.85,
      "forecast_accuracy": 0.3,
      "volume_of_error": 0.7,
      "pct_error": 0.4,
      "weeks_affected": 6,
      "prior_anomaly_count": 2,
      "dc_region_encoded": 1,
      "product_lifecycle_encoded": 1,
      "is_seasonal_encoded": 0
    }}'""")


if __name__ == "__main__":
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_ML_WORKSPACE"],
    )

    deploy(ml_client, credential)
