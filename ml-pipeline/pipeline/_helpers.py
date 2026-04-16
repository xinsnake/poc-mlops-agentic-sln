"""
_helpers.py

Shared helper functions for run_pipeline.py and run_sweep.py.

Centralises: MLClient setup, compute management, data asset resolution,
metric gates, model registration, and endpoint deployment.
"""

import os
import sys
from dotenv import load_dotenv
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    AmlCompute,
    Model,
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    CodeConfiguration,
)
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

COMPUTE_NAME = os.environ.get("AML_COMPUTE_NAME", "spike-cluster")
COMPUTE_SIZE = os.environ.get("AML_COMPUTE_SIZE", "Standard_DS2_v2")
MODEL_NAME = os.environ.get("AML_MODEL_NAME", "anomaly-classifier")
ENDPOINT_NAME = os.environ.get("AML_ENDPOINT_NAME", "anomaly-classifier-endpoint")
DEPLOYMENT_NAME = "blue"
F1_THRESHOLD = 0.75


# ── MLClient ─────────────────────────────────────────────────────────────────

def get_ml_client() -> MLClient:
    """Create an MLClient using DefaultAzureCredential."""
    credential = DefaultAzureCredential()
    return MLClient(
        credential=credential,
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_ML_WORKSPACE"],
    )


# ── Compute ──────────────────────────────────────────────────────────────────

def get_or_create_compute(ml_client: MLClient) -> str:
    """Get existing compute cluster or create a new one (auto-scales to 0)."""
    try:
        ml_client.compute.get(COMPUTE_NAME)
        print(f"Using existing compute cluster: {COMPUTE_NAME}")
    except Exception:
        print(f"Creating compute cluster: {COMPUTE_NAME} ({COMPUTE_SIZE})")
        compute = AmlCompute(
            name=COMPUTE_NAME,
            size=COMPUTE_SIZE,
            min_instances=0,
            max_instances=1,
        )
        ml_client.compute.begin_create_or_update(compute).result()
        print(f"Compute cluster created: {COMPUTE_NAME}")
    return COMPUTE_NAME


# ── Data Assets ──────────────────────────────────────────────────────────────

def resolve_data_assets(ml_client: MLClient) -> tuple:
    """
    Resolve latest versions of training and validation data assets.
    Returns (train_version, val_version) as strings.
    """
    train_version = ml_client.data.get(name="anomaly-train", label="latest").version
    val_version = ml_client.data.get(name="anomaly-validation", label="latest").version
    print(f"Using data assets: anomaly-train:{train_version}, anomaly-validation:{val_version}")
    return train_version, val_version


# ── Metric Gates ─────────────────────────────────────────────────────────────

def metric_gate(f1: float) -> bool:
    """
    Minimum quality gate: F1_weighted must meet threshold.
    Returns True if the model passes, False if it should be rejected.
    """
    if f1 >= F1_THRESHOLD:
        print(f"\n✅ Metric gate PASSED: F1_weighted={f1:.4f} >= {F1_THRESHOLD}")
        return True
    else:
        print(f"\n❌ Metric gate FAILED: F1_weighted={f1:.4f} < {F1_THRESHOLD}")
        print("   Model will NOT be registered. Improve training data or hyperparameters.")
        return False


def champion_challenger_gate(ml_client: MLClient, new_f1: float) -> bool:
    """
    Champion/challenger gate: only deploy if new model beats production.
    Returns True if new model should be deployed.
    On first run (no endpoint exists), returns True automatically.
    """
    try:
        endpoint = ml_client.online_endpoints.get(ENDPOINT_NAME)
    except Exception:
        print(f"\n✅ Champion/challenger gate PASSED: no existing endpoint '{ENDPOINT_NAME}' — first deployment")
        return True

    try:
        traffic = endpoint.traffic or {}
        active_deployment = next((name for name, pct in traffic.items() if pct > 0), None)
        if not active_deployment:
            print(f"\n✅ Champion/challenger gate PASSED: endpoint exists but has no active deployment")
            return True

        deployment = ml_client.online_deployments.get(
            name=active_deployment,
            endpoint_name=ENDPOINT_NAME,
        )
        current_model_ref = deployment.model
        parts = current_model_ref.split(":")
        current_model = ml_client.models.get(name=parts[1], version=parts[2])

        # Read F1 from model tags (set during registration)
        current_f1_str = current_model.tags.get("f1_weighted")
        if not current_f1_str:
            print(f"\n✅ Champion/challenger gate PASSED: current model has no f1_weighted tag — cannot compare")
            return True

        current_f1 = float(current_f1_str)

        if new_f1 > current_f1:
            print(f"\n✅ Champion/challenger gate PASSED: new F1={new_f1:.4f} > current F1={current_f1:.4f}")
            return True
        else:
            print(f"\n❌ Champion/challenger gate FAILED: new F1={new_f1:.4f} <= current F1={current_f1:.4f}")
            print("   Current production model is equal or better. Skipping deployment.")
            return False

    except Exception as e:
        print(f"\n⚠️  Champion/challenger gate: error comparing models ({e}). Proceeding with deployment.")
        return True


# ── Model Registration ───────────────────────────────────────────────────────

def register_model(ml_client: MLClient, training_job_name: str, f1: float) -> str:
    """
    Register the trained model in the AML Model Registry.
    Returns the registered model version string.
    """
    # Pipeline jobs use named outputs; standalone command jobs use artifacts/outputs
    model_path = f"azureml://jobs/{training_job_name}/outputs/model_output/paths/model.txt"

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
            "f1_weighted": f"{f1:.4f}",
            "spike": "true",
        },
    )

    registered = ml_client.models.create_or_update(model)
    print(f"\nModel registered: {registered.name}:{registered.version}")
    print(f"  ID: {registered.id}")
    _print_aml_model_url(registered)
    return registered.version


# ── Deployment ───────────────────────────────────────────────────────────────

def deploy_model(ml_client: MLClient) -> None:
    """
    Deploy the latest registered model to the Managed Online Endpoint.
    Handles: endpoint creation, score.py upload, deployment, traffic routing.
    """
    # Step 1: Create/update endpoint
    print("\nDeploying: creating endpoint...")
    endpoint = ManagedOnlineEndpoint(
        name=ENDPOINT_NAME,
        description="Forecast anomaly classifier — UC4 spike PoC",
        auth_mode="key",
    )
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"  Endpoint ready: {ENDPOINT_NAME}")

    # Step 2: Upload score.py (bypasses SAS restriction)
    print("Deploying: uploading score.py...")
    code_ref = _upload_and_register_score_script(ml_client)

    # Step 3: Create deployment
    model_version = ml_client.models.get(name=MODEL_NAME, label="latest").version
    print(f"Deploying: creating deployment with {MODEL_NAME}:{model_version} (~10 minutes)...")
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

    # Step 4: Route traffic
    print("Deploying: routing traffic...")
    endpoint.traffic = {DEPLOYMENT_NAME: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print("  Traffic routed: 100% → blue")

    # Step 5: Print details
    endpoint = ml_client.online_endpoints.get(ENDPOINT_NAME)
    keys = ml_client.online_endpoints.get_keys(ENDPOINT_NAME)
    print(f"\n{'='*60}")
    print(f"Endpoint deployed successfully!")
    print(f"  URL:     {endpoint.scoring_uri}")
    print(f"  API Key: {keys.primary_key}")


def _upload_and_register_score_script(ml_client: MLClient) -> str:
    """Upload score.py via OAuth (bypasses SAS restriction), register as Code asset."""
    from azure.ai.ml.entities._assets._artifacts.code import Code

    datastore = ml_client.datastores.get_default()
    blob_base_url = f"https://{datastore.account_name}.blob.core.windows.net"
    blob_service = BlobServiceClient(account_url=blob_base_url, credential=DefaultAzureCredential())
    container_client = blob_service.get_container_client(datastore.container_name)

    score_path = os.path.join(os.path.dirname(__file__), "..", "src", "score.py")

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


# ── URL Helpers ──────────────────────────────────────────────────────────────

def get_aml_studio_run_url(job_name: str) -> str:
    sub = os.environ["AZURE_SUBSCRIPTION_ID"]
    rg = os.environ["AZURE_RESOURCE_GROUP"]
    ws = os.environ["AZURE_ML_WORKSPACE"]
    return f"https://ml.azure.com/runs/{job_name}?wsid=/subscriptions/{sub}/resourcegroups/{rg}/workspaces/{ws}"


# ── Code Upload (OAuth workaround) ──────────────────────────────────────────

def upload_and_register_code(ml_client: MLClient, local_dir: str, asset_name: str) -> str:
    """
    Upload a local directory to blob storage via OAuth, register as an AML Code asset.
    Returns the azureml:name:version reference.

    This bypasses the SAS token restriction — the storage account has
    allowSharedKeyAccess: false, so the SDK's built-in code upload fails.
    """
    from azure.ai.ml.entities._assets._artifacts.code import Code

    datastore = ml_client.datastores.get_default()
    blob_base_url = f"https://{datastore.account_name}.blob.core.windows.net"
    blob_service = BlobServiceClient(account_url=blob_base_url, credential=DefaultAzureCredential())
    container_client = blob_service.get_container_client(datastore.container_name)

    # Probe for next available version
    next_version = "1"
    for candidate in range(50, 0, -1):
        try:
            ml_client._code.get(name=asset_name, version=str(candidate))
            next_version = str(candidate + 1)
            break
        except Exception:
            continue

    # Upload all files in the directory
    for root, _dirs, files in os.walk(local_dir):
        for filename in files:
            if filename.startswith(".") or filename.endswith(".pyc") or "__pycache__" in root:
                continue
            local_path = os.path.join(root, filename)
            rel_path = os.path.relpath(local_path, local_dir)
            blob_name = f"spike-code/{asset_name}/v{next_version}/{rel_path}"
            with open(local_path, "rb") as f:
                container_client.upload_blob(name=blob_name, data=f, overwrite=True)
            print(f"  Uploaded {rel_path} → blob: {blob_name}")

    blob_uri = f"{blob_base_url}/{datastore.container_name}/spike-code/{asset_name}/v{next_version}/"
    code = Code(name=asset_name, version=next_version, path=blob_uri)
    result = ml_client._code.create_or_update(code)
    print(f"  Registered Code asset: {result.name}:{result.version}")

    return f"azureml:{result.name}:{result.version}"


def _print_aml_model_url(model) -> None:
    sub = os.environ["AZURE_SUBSCRIPTION_ID"]
    rg = os.environ["AZURE_RESOURCE_GROUP"]
    ws = os.environ["AZURE_ML_WORKSPACE"]
    print(f"  https://ml.azure.com/models/{model.name}/version/{model.version}"
          f"?wsid=/subscriptions/{sub}/resourcegroups/{rg}/workspaces/{ws}")
