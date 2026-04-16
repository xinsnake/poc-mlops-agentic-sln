# ML Pipeline Spike — Implementation Summary

**Date:** 2026-04-10 (updated 2026-04-16)
**Scope:** UC4 Forecast Parameter Agent — ML pipeline proof of concept
**Location:** `ml-pipeline/`

---

## What was built

An end-to-end ML pipeline proving that forecast anomaly records can flow from Snowflake through Azure ML and be served as live predictions via a REST endpoint — with automated quality gates.

### Original flow (manual, step-by-step)

```
01-generate_data.py → 02-data_pipeline.py → 03-submit_training.py → 04-register_model.py → 05-deploy_endpoint.py
```

### Current flow (combined pipeline with gates)

```
01-generate_data.py → 02-data_pipeline.py → run_pipeline.py
                                                 ↓
                                            Submit training to AML
                                                 ↓
                                            METRIC GATE: F1 ≥ 0.80?
                                                 ↓ yes
                                            Register model
                                                 ↓
                                            CHAMPION/CHALLENGER GATE: new F1 > production?
                                                 ↓ yes
                                            Deploy to endpoint
```

Steps 01 and 02 still run locally. `run_pipeline.py` handles everything from training through deployment in one command, with automated gates that prevent bad models from going live.

---

## Pipeline scripts

| Script | Runs on | What it does |
|--------|---------|-------------|
| `pipeline/01-generate_data.py` | Local | Generates 800 synthetic anomaly rows, inserts into Snowflake |
| `pipeline/02-data_pipeline.py` | Local | Extracts from Snowflake, time-splits 60/40, uploads to AML blob, registers Data Assets |
| `pipeline/run_pipeline.py` | Local (submits to AML) | **Combined pipeline:** trains → gates → registers → deploys (see below) |
| `pipeline/run_sweep.py` | Local (submits to AML) | Hyperparameter sweep variant — tries multiple configs, picks best (Phase 2) |
| `pipeline/_helpers.py` | Local | Shared helpers: compute, gates, registration, deployment |
| `pipeline/03-submit_training.py` | Local (submits to AML) | *Legacy* — standalone Command Job (still works for quick one-off training) |
| `pipeline/04-register_model.py` | Local | *Legacy* — standalone model registration |
| `pipeline/05-deploy_endpoint.py` | Local (deploys to AML) | *Legacy* — standalone endpoint deployment |
| `src/train.py` | AML cluster | LightGBM training, MLflow metric logging, saves model to outputs |
| `src/score.py` | AML endpoint | Serves predictions via init() + run() |

---

## How `run_pipeline.py` works

```
run_pipeline.py
  │
  ├─ 1. Resolve latest data assets (anomaly-train, anomaly-validation)
  ├─ 2. Get/create compute cluster (auto-scales to 0 when idle)
  ├─ 3. Submit training as AML Pipeline Job
  │        └─ train.py runs on AML compute:
  │           ├─ Trains LightGBM classifier
  │           ├─ Logs metrics via MLflow (accuracy, F1, AUC)
  │           ├─ Logs hyperparameters via MLflow
  │           └─ Saves model.txt to outputs
  │
  ├─ 4. METRIC GATE: Is F1_weighted ≥ 0.80?
  │        ├─ NO  → "Model will NOT be registered", exit code 1
  │        └─ YES ↓
  │
  ├─ 5. Register model in AML Model Registry (tags: f1, training job, framework)
  │
  ├─ 6. CHAMPION/CHALLENGER GATE: Is new F1 > current production model's F1?
  │        ├─ NO  → "Current model is better, skipping deploy", exit code 0
  │        ├─ FIRST RUN (no endpoint) → skip gate, deploy directly
  │        └─ YES ↓
  │
  └─ 7. Deploy to Managed Online Endpoint (blue deployment, 100% traffic)
```

### Metric gates

| Gate | When | Logic | Failure action |
|------|------|-------|----------------|
| **Minimum quality** | After training, before registration | `F1_weighted ≥ 0.80` | Stop — model not registered |
| **Champion/challenger** | After registration, before deployment | `new F1 > production F1` | Stop — model registered but not deployed |

The champion/challenger gate handles edge cases gracefully:
- **First run** (no endpoint exists): skips the gate and deploys directly
- **No active deployment**: skips the gate
- **Can't read production metrics**: proceeds with deployment (warns)

---

## Key technical decisions

### Pipeline Job (upgraded from Command Job)
Originally used `command()` for simplicity. Now uses `@dsl.pipeline` with a reusable training component. Benefits:
- Component is reusable across `run_pipeline.py` and `run_sweep.py`
- Pipeline graph is visible in AML Studio
- Outputs are typed and chainable

### MLflow metric logging (upgraded from print)
`train.py` now logs metrics via `mlflow.log_metric()` and hyperparameters via `mlflow.log_param()`. This is required for:
- `run_pipeline.py` to read metrics for gating decisions
- `run_sweep.py` to compare trials by primary metric
- AML Studio metric visualisation

### Combined pipeline with gates
`run_pipeline.py` combines train + register + deploy in one command. The metric gates provide the safety that separate scripts would otherwise give:
- Bad model? Gate stops registration
- Worse than production? Gate stops deployment
- Both gates pass? Full automated flow

### Why step 02 stays local
The data pipeline connects to Snowflake using credentials from `.env`. AML compute clusters don't have those credentials. Including step 02 in the AML pipeline would require Key Vault + Managed Identity wiring — deferred to production.

### LightGBM for classification
Chosen for speed, memory efficiency, and strong performance on tabular data with class imbalance. `class_weight="balanced"` applied for 70/30 split. See `docs/plan/20260320-spike-plan.md` for alternatives considered.

### Storage auth workaround
Storage account has `allowSharedKeyAccess: false` enforced by Azure Policy. AML SDK uses SAS tokens internally for code uploads, which are blocked. Workaround: upload `score.py` directly via `BlobServiceClient` with `DefaultAzureCredential` (OAuth), register as a Code asset, reference by name.

### Realistic training data
`01-generate_data.py` uses a 15% noise rate — features are sampled from the opposite class distribution with this probability, creating realistic overlap.

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
- `ml_client._code` is a private API — fragile across SDK versions (used in deployment helper)
- No CI/CD trigger — pipeline is run manually from terminal
- No prediction logging — outputs are not persisted back to Snowflake
- Step 02 (data pipeline) runs locally — not inside AML pipeline (Snowflake creds not in Key Vault)

---

## What's next

- ~~Convert to AML Pipeline Job with reusable components~~ ✅ Done (run_pipeline.py)
- ~~Add metric gates (minimum quality + champion/challenger)~~ ✅ Done
- ~~MLflow metric logging~~ ✅ Done
- Hyperparameter sweep (run_sweep.py) — Phase 2
- Feature importance logging — Phase 3
- Build UC4 agentic layer: reads classification from endpoint, reasons about parameter change, presents recommendation to demand planner
- Build UC1 Forecast Exception Agent
- Replace synthetic data with real Blue Yonder / Snowflake data when landing zone is available
