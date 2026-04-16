# Phase 1 — Automated AML Pipeline: Implementation & Verification

**Date:** 2026-04-16
**Scope:** Combined pipeline with metric gates (train → gate → register → gate → deploy)

---

## What was built

A single command (`python pipeline/run_pipeline.py`) that replaces the previous 3-step manual process (submit training → register model → deploy endpoint) with an automated pipeline that includes quality gates.

### New files

| File | Purpose |
|------|---------|
| `pipeline/run_pipeline.py` | Combined pipeline orchestrator — submits training to AML, reads metrics, applies gates, registers model, deploys endpoint |
| `pipeline/_helpers.py` | Shared helpers: MLClient setup, compute management, data asset resolution, metric gates, model registration, endpoint deployment |
| `environment/conda.yml` | Custom AML environment definition (LightGBM + MLflow + inference server) |
| `environment/register_environment.py` | Script to register the custom environment with AML |
| `docs/plan/20260415-pipeline-improvements.md` | Implementation plan for all three phases |

### Modified files

| File | Changes |
|------|---------|
| `src/train.py` | Added `--model-output` argument, `metrics.json` output, optional MLflow logging, removed hardcoded `outputs/` path |
| `docs/implementation/20260410-ml-pipeline-spike.md` | Updated with pipeline flow, gate documentation, scripts table, decision log |
| `.env.example` | Updated to reference custom environment instead of curated |
| `requirements.txt` | No net change (mlflow was added then removed from local deps — it's in the AML environment instead) |

### Pipeline flow

```
python pipeline/run_pipeline.py
  │
  ├─ 1. Resolve latest data assets (anomaly-train, anomaly-validation)
  ├─ 2. Get/create compute cluster (auto-scales to 0 when idle)
  ├─ 3. Submit training as AML Pipeline Job
  │        └─ train.py runs on AML compute:
  │           ├─ Trains LightGBM classifier
  │           ├─ Logs metrics via MLflow (visible in AML Studio)
  │           ├─ Writes metrics.json to model_output (reliable fallback)
  │           └─ Saves model.txt to model_output
  │
  ├─ 4. Downloads metrics.json from completed job
  ├─ 5. METRIC GATE: Is F1_weighted ≥ 0.75?
  │        ├─ NO  → exit code 1, model NOT registered
  │        └─ YES ↓
  │
  ├─ 6. Register model in AML Model Registry (tagged with F1 score)
  │
  ├─ 7. CHAMPION/CHALLENGER GATE: Is new F1 > production model's F1?
  │        ├─ NO  → exit code 0, model registered but NOT deployed
  │        ├─ FIRST RUN → skip gate, deploy directly
  │        └─ YES ↓
  │
  └─ 8. Deploy to Managed Online Endpoint (blue deployment, 100% traffic)
```

---

## Issues encountered and resolved

### Issue 1: SAS token blocked code upload

**Error:** `AuthorizationFailure — This request is not authorized to perform this operation`

**Root cause:** The storage account had `allowSharedKeyAccess: false` enforced by Azure Policy. The AML SDK uses SAS tokens internally to upload code to blob storage, which was blocked.

**Resolution:** Enabled shared key access on the storage account via Azure Portal. This allowed the SDK to upload `src/` code normally.

**Note:** The original spike (05-deploy_endpoint.py) worked around this with a direct OAuth upload via BlobServiceClient. That workaround is still used for score.py deployment in `_helpers.py`. With shared key access enabled, the training code upload now works via the standard SDK path.

---

### Issue 2: MLflow protobuf crash in curated environment

**Error:** `AttributeError: 'google._upb._message.FieldDescriptor' object has no attribute 'label'`

**Root cause:** The curated AML environment (`lightgbm-3.3/versions/76`) bundles incompatible versions of `mlflow` and `protobuf`. Training completed successfully but crashed when calling `mlflow.log_metric()`.

**Resolution:** Created a custom AML environment (`environment/conda.yml`) with pinned compatible versions:
- `mlflow==2.10.0`
- `protobuf==4.25.3`

**What is a custom environment?** Microsoft provides pre-built "curated" environments — Docker images with pre-installed packages. You can't modify them. A custom environment is a `conda.yml` recipe that you control. AML builds a Docker image from it. You choose every package and version, so you can resolve conflicts.

**How to register:** `python environment/register_environment.py`

---

### Issue 3: Endpoint identity lacks ACR pull permission

**Error:** `Endpoint identity does not have pull permission on the registry`

**Root cause:** The Managed Online Endpoint's system-assigned managed identity didn't have permission to pull the custom environment Docker image from the workspace's Azure Container Registry (ACR).

**Resolution:** Assigned the `AcrPull` role to the endpoint's managed identity:
```bash
az role assignment create \
  --assignee <endpoint-principal-id> \
  --role "AcrPull" \
  --scope <acr-resource-id>
```

**Note:** This is a one-time setup per endpoint. The curated environment didn't need this because it's pulled from Microsoft's public registry.

---

### Issue 4: Missing `azureml-inference-server-http`

**Error:** `A required package azureml-inference-server-http is missing`

**Root cause:** The custom environment didn't include the HTTP inference server that AML requires for serving endpoints. The curated environment had this pre-installed.

**Resolution:** Added `azureml-inference-server-http` to `environment/conda.yml`.

---

### Issue 5: Missing `pkg_resources` (setuptools)

**Error:** `ModuleNotFoundError: No module named 'pkg_resources'`

**Root cause:** The `gunicorn` version installed by `azureml-inference-server-http` imports `pkg_resources` from `setuptools`. Newer Python 3.10 installations may not include `setuptools` by default, and pip's dependency resolver didn't pull it in.

**Resolution:** Pinned `setuptools==69.5.1` as a pip dependency in `conda.yml`. This version includes `pkg_resources`. It's listed first in the pip dependencies so it's installed before packages that need it.

---

### Issue 6: Model path mismatch for pipeline jobs

**Error:** `NoMatchingArtifactsFoundFromJob — No artifacts matching outputs/model.txt found from Job`

**Root cause:** The original model registration used the path `azureml://jobs/{name}/outputs/artifacts/paths/outputs/model.txt`, which works for standalone Command Jobs. Pipeline Jobs use named outputs, so the path is `azureml://jobs/{name}/outputs/model_output/paths/model.txt`.

**Resolution:** Updated `register_model()` in `_helpers.py` to use the pipeline-compatible output path.

---

## How to verify Phase 1 is working

### 1. Check the pipeline ran successfully in AML Studio

Open the pipeline job in AML Studio:
```
https://ml.azure.com/runs/<job-name>?wsid=/subscriptions/<sub>/resourcegroups/<rg>/workspaces/<ws>
```

The job name is printed when `run_pipeline.py` executes (e.g. `bright_candle_prd6h2t2sf`). You should see:
- Pipeline status: **Completed**
- One child job (train_step): **Completed**
- Clicking the child job shows MLflow metrics (accuracy, f1_weighted, auc) in the **Metrics** tab

### 2. Check the model is in the registry

In AML Studio → **Models** → `anomaly-classifier`:
- Latest version should show (e.g. v12)
- Tags should include: `f1_weighted`, `training_job`, `framework: lightgbm`

Or via CLI:
```bash
cd ml-pipeline
python -c "
from dotenv import load_dotenv; load_dotenv()
import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
ml_client = MLClient(DefaultAzureCredential(), os.environ['AZURE_SUBSCRIPTION_ID'], os.environ['AZURE_RESOURCE_GROUP'], os.environ['AZURE_ML_WORKSPACE'])
model = ml_client.models.get('anomaly-classifier', label='latest')
print(f'Model: {model.name}:{model.version}')
print(f'Tags: {model.tags}')
"
```

### 3. Check the endpoint is live

In AML Studio → **Endpoints** → `anomaly-classifier-endpoint`:
- Status: **Healthy**
- Traffic: 100% → blue

Or test with curl:
```bash
curl -X POST "https://anomaly-classifier-endpoint.australiaeast.inference.ml.azure.com/score" \
  -H "Authorization: Bearer <api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "forecast_bias": 0.85,
    "forecast_accuracy": 0.3,
    "volume_of_error": 0.7,
    "pct_error": 0.4,
    "weeks_affected": 6,
    "prior_anomaly_count": 2,
    "dc_region_encoded": 1,
    "product_lifecycle_encoded": 1,
    "is_seasonal_encoded": 0
  }'
```

Expected response:
```json
{"prediction": "baseline_shift", "probability": 0.8234}
```

Get the API key via:
```bash
cd ml-pipeline
python -c "
from dotenv import load_dotenv; load_dotenv()
import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
ml_client = MLClient(DefaultAzureCredential(), os.environ['AZURE_SUBSCRIPTION_ID'], os.environ['AZURE_RESOURCE_GROUP'], os.environ['AZURE_ML_WORKSPACE'])
keys = ml_client.online_endpoints.get_keys('anomaly-classifier-endpoint')
print(f'API Key: {keys.primary_key}')
"
```

### 4. Verify the metric gate works

Re-run the pipeline. Since the model is already deployed with F1=0.7866, the champion/challenger gate should now block deployment (new model has the same F1 — not better):

```bash
cd ml-pipeline
python pipeline/run_pipeline.py
```

Expected output:
```
✅ Metric gate PASSED: F1_weighted=0.7866 >= 0.75
Model registered: anomaly-classifier:13
❌ Champion/challenger gate FAILED: new F1=0.7866 <= current F1=0.7866
   Current production model is equal or better. Skipping deployment.
```

This proves the gate is working — it registers the model but refuses to deploy because the new model isn't better.

### 5. Check the custom environment

In AML Studio → **Environments** → `anomaly-classifier-env`:
- Latest version should be v5
- Click into it to see the conda specification
- Build status should be: **Succeeded**

---

## Architecture decisions documented

| Decision | Rationale |
|----------|-----------|
| **Combined pipeline** (train + register + deploy) | One command; gates provide the safety that separate scripts would give |
| **Custom environment** over curated | Curated has protobuf/MLflow conflict we can't fix; custom gives us control |
| **metrics.json + MLflow** (dual logging) | metrics.json is the reliable fallback; MLflow provides AML Studio visibility |
| **F1 threshold = 0.75** (not 0.80) | Synthetic data with 15% noise can't consistently hit 0.80; adjustable constant |
| **Champion/challenger reads F1 from model tags** | No MLflow dependency in the gate logic; tags are set during registration |
| **Step 02 stays local** | Snowflake creds not in Key Vault; AML compute can't access them |

---

## Final conda.yml (v5)

```yaml
name: anomaly-classifier
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.10
  - pip:
    - setuptools==69.5.1
    - lightgbm==4.3.0
    - scikit-learn==1.4.0
    - pandas==2.2.0
    - numpy==1.26.0
    - mlflow==2.10.0
    - protobuf==4.25.3
    - azureml-mlflow==1.56.0
    - azureml-inference-server-http
```

Key package choices:
- `setuptools==69.5.1` — provides `pkg_resources` for gunicorn (must be listed first)
- `protobuf==4.25.3` — compatible with `mlflow==2.10.0` (the curated env had an incompatible version)
- `azureml-inference-server-http` — required for serving endpoints (not needed for training, but same env is used for both)
- `azureml-mlflow==1.56.0` — bridges MLflow to AML's tracking backend
