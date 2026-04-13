# ML Pipeline Spike — Implementation Summary

**Date:** 2026-04-10
**Scope:** UC4 Forecast Parameter Agent — ML pipeline proof of concept
**Location:** `work/spike/`

---

## What was built

An end-to-end ML pipeline proving that forecast anomaly records can flow from Snowflake through Azure ML and be served as live predictions via a REST endpoint.

```
Snowflake (anomaly_records)
    ↓ 02-data_pipeline.py
Azure Blob Storage (train.csv / validation.csv)
    ↓ registered as AML Data Assets
AML Compute Cluster (spike-cluster)
    ↓ 03-submit_training.py → train.py
LightGBM model (model.txt)
    ↓ 04-register_model.py
AML Model Registry (anomaly-classifier:N)
    ↓ 05-deploy_endpoint.py
AML Managed Online Endpoint
    ↓ score.py
{"prediction": "baseline_shift", "probability": 0.89}
```

---

## Pipeline scripts

| Script | Runs on | What it does |
|--------|---------|-------------|
| `pipeline/01-generate_data.py` | Local | Generates 800 synthetic anomaly rows, inserts into Snowflake |
| `pipeline/02-data_pipeline.py` | Local | Extracts from Snowflake, time-splits 60/40, uploads to AML blob, registers Data Assets |
| `pipeline/03-submit_training.py` | Local (submits to AML) | Submits LightGBM Command Job to spike-cluster. Auto-resolves latest data asset version |
| `pipeline/04-register_model.py` | Local | Auto-detects latest completed training job, registers model in Model Registry |
| `pipeline/05-deploy_endpoint.py` | Local (deploys to AML) | Uploads score.py, registers Code asset, creates/updates Managed Online Endpoint |
| `src/train.py` | AML cluster | LightGBM training, evaluation, saves model.txt to outputs |
| `src/score.py` | AML endpoint | Serves predictions via init() + run() |

---

## Key technical decisions

### LightGBM for classification
Chosen for speed, memory efficiency, and strong performance on tabular data with class imbalance. `class_weight="balanced"` applied for 70/30 split. See `work/spike-plan.md` for alternatives considered.

### Command Job (not AML Pipeline)
Used `command()` job type for simplicity in the spike. Production upgrade path is `@pipeline` with reusable components. Not over-engineered for PoC.

### Storage auth workaround
Storage account has `allowSharedKeyAccess: false` enforced by `Azure_Security_Baseline` policy at management group level — cannot be changed. AML SDK uses SAS tokens internally for code uploads, which are blocked. Workaround in `05-deploy_endpoint.py`: upload `score.py` directly via `BlobServiceClient` with `DefaultAzureCredential` (OAuth), register as a Code asset, reference by name.

### Version auto-resolution
- Data assets: resolved via `label="latest"` in `03-submit_training.py`
- Model: resolved via `label="latest"` in `05-deploy_endpoint.py`
- Code asset: probed via `_code.get()` scan (private API has no `list()` method)
- Code blob: uploaded to version-specific path (`spike-code/v{N}/score.py`) so AML always snapshots fresh content

### Realistic training data
`01-generate_data.py` uses a 15% noise rate (`NOISE_RATE=0.15`) — features are sampled from the opposite class distribution with this probability, creating realistic overlap. This produces probabilities in the `0.05–0.89` range rather than `0.0/1.0` for all inputs.

---

## Azure resources created

| Resource | Name |
|----------|------|
| AML Workspace | `<your-aml-workspace>` |
| Compute Cluster | `spike-cluster` (Standard_DS2_v2, SystemAssigned MI) |
| Storage Account | `<your-storage-account>` |
| Managed Online Endpoint | `anomaly-classifier-endpoint` |
| Model Registry | `anomaly-classifier` (versioned) |
| Snowflake Warehouse | `COMPUTE_WH` / `SPIKE_DB.PUBLIC.anomaly_records` |

---

## Environment variables (.env)

```
SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD
SNOWFLAKE_WAREHOUSE, SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA
AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_ML_WORKSPACE
AML_COMPUTE_NAME, AML_COMPUTE_SIZE
AML_ENVIRONMENT
AML_ENDPOINT_NAME, AML_MODEL_NAME
```

---

## Known limitations (spike scope)

- `01-generate_data.py` is synthetic — in production, Blue Yonder writes to Snowflake directly
- Metrics are logged via `print()` not MLflow (protobuf version conflict in curated environment)
- `ml_client._code` is a private API — fragile across SDK versions
- No CI/CD trigger — pipeline is run manually step by step
- No prediction logging — outputs are not persisted back to Snowflake

---

## What's next

- Convert to AML Pipeline Job with reusable components
- Build UC4 agentic layer: reads classification from endpoint, reasons about parameter change, presents recommendation to demand planner
- Build UC1 Forecast Exception Agent
- Replace synthetic data with real Blue Yonder / Snowflake data when landing zone is available
