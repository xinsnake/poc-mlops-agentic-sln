# Implementation Plan — Pipeline Job, Feature Importance & Sweep

**Date:** 2026-04-15 (updated 2026-04-16)
**Scope:** Three enhancements to the ML pipeline

---

## Decisions Made

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | **`run_pipeline.py` is combined** (train + register + deploy) — not separate scripts | One command runs the full flow. Metric gates provide the safety that modular scripts would otherwise give. |
| 2 | **Metric gate before registration**: F1 ≥ 0.80 | Prevents bad models from entering the registry. 0.80 matches the existing spike target. |
| 3 | **Champion/challenger gate before deployment**: new model F1 > current production F1 | Prevents deploying a model that's worse than what's already live. |
| 4 | **Step 02 (data pipeline) stays local** — not inside the AML pipeline | Snowflake credentials live in local `.env`. AML compute doesn't have access. Would need Key Vault + Managed Identity wiring to include it — deferred to production. |
| 5 | **`03-submit_training.py` kept** as a simpler standalone alternative | Still works for quick one-off training without the full pipeline flow. |
| 6 | **MLflow replaces `print()`-based metric logging** | Required by Sweep Job to read primary metric across trials. Also better practice for AML Studio visibility. |
| 7 | **Feature importance logged as JSON artifact + stdout table** | Enables explainability — stakeholders can see why predictions are made. |
| 8 | **Sweep uses random sampling, 20 trials, BanditPolicy** | Good exploration vs cost balance. BanditPolicy kills poor runs early. |

---

## Problem

The current pipeline uses a standalone Command Job (`03-submit_training.py`) with hardcoded hyperparameters and no model explainability. Three improvements are needed:

1. **Log feature importance** — LightGBM calculates feature importance internally but `train.py` discards it
2. **Convert to AML Pipeline Job** — replace sequential manual scripts with a combined automated pipeline
3. **Hyperparameter sweep** — use AML Sweep Job to automatically find optimal hyperparameters

---

## How the pipeline runs after this work

### Before (current — all manual)

```
You run locally:  01 → 02 → 03 → 04 → 05
                  ↑     ↑     ↑     ↑     ↑
               gen data  data  submit  register  deploy
                        pipeline training model   endpoint
```

### After (combined pipeline with gates)

```
You run locally:  01 → 02 → run_pipeline.py
                  ↑     ↑         ↑
               gen data  data   trains + gates + registers + deploys
                        pipeline (one command does it all)
```

### What `run_pipeline.py` does internally

```
run_pipeline.py
  │
  ├─ 1. Resolve latest data assets (anomaly-train, anomaly-validation)
  ├─ 2. Get/create compute cluster
  ├─ 3. Submit training Pipeline Job to AML
  │        └─ train.py runs on AML compute
  │           ├─ Trains LightGBM
  │           ├─ Logs metrics via MLflow (accuracy, F1, AUC)
  │           ├─ Logs feature importance (JSON artifact + table)
  │           └─ Saves model to outputs/
  │
  ├─ 4. METRIC GATE: Is F1_weighted ≥ 0.80?
  │        ├─ NO  → log warning, exit. Model not registered.
  │        └─ YES ↓
  │
  ├─ 5. Register model in AML Model Registry
  │
  ├─ 6. CHAMPION/CHALLENGER GATE: Is new model F1 > current production F1?
  │        ├─ NO  → log "current model is better, skipping deploy", exit.
  │        └─ YES ↓
  │
  └─ 7. Deploy to Managed Online Endpoint (blue/green swap)
```

### Why step 02 isn't inside the AML pipeline

The data pipeline connects to Snowflake using username/password from your local `.env`. AML compute clusters don't have those credentials. To include step 02, you'd need:
1. Store Snowflake creds in Azure Key Vault
2. Grant AML compute's Managed Identity access
3. Inject as environment variables into the pipeline step

This is solvable but adds complexity beyond spike scope. In production, you'd either do the above or use Snowflake OAuth via Entra ID.

---

## Approach

All three changes share a dependency: `train.py` needs MLflow metric logging (currently uses `print()`). The sweep job *requires* MLflow to read the primary metric across trials. This is the first thing to address.

### Change map

```
train.py (enhanced)
  ├─ Feature importance logging (JSON artifact + stdout table)
  ├─ MLflow metric logging (replaces print-based logging)
  └─ --model-output argument (explicit output path for pipeline chaining)

pipeline/run_pipeline.py (new — combined)
  ├─ Defines training component via command()
  ├─ Submits training Pipeline Job to AML
  ├─ Metric gate (F1 ≥ 0.80)
  ├─ Registers model in Model Registry
  ├─ Champion/challenger gate (new F1 > production F1)
  └─ Deploys to Managed Online Endpoint

pipeline/run_sweep.py (new)
  ├─ Reuses the same training component
  ├─ Wraps in .sweep() with search space
  ├─ Same gates as run_pipeline.py (metric + champion/challenger)
  └─ Registers + deploys best trial's model
```

### What stays unchanged
- `01-generate_data.py` — synthetic data gen
- `02-data_pipeline.py` — Snowflake extraction + processing + AML data asset registration
- `05-deploy_endpoint.py` — kept as standalone deploy tool (useful for re-deploying without retraining)
- `src/score.py` — scoring script
- `03-submit_training.py` — kept as simpler training-only alternative

---

## Phases

### Phase 1 — Automated AML Pipeline
*Combined pipeline: train + metric gates + register + deploy in one command*

#### 1a. mlflow-logging
**Add MLflow metric logging to `src/train.py`**

- Add `mlflow` import and auto-logging setup
- Replace `print("##[metric]...")` with `mlflow.log_metric()` (keep print for readability)
- Add `--model-output` CLI argument (defaults to `outputs/` for backward compatibility)
- Save model to `{model_output}/model.txt` instead of hardcoded `outputs/model.txt`
- Update `requirements.txt` to include `mlflow>=2.10.0`

Risk: The curated LightGBM environment had a protobuf conflict with MLflow. If it persists, we may need a custom AML Environment definition.

#### 1b. extract-shared-helpers
**Create `pipeline/_helpers.py` — shared logic for pipeline scripts**

Both `run_pipeline.py` (this phase) and `run_sweep.py` (phase 2) need:
- `get_or_create_compute()` — compute cluster management
- `get_ml_client()` — MLClient instantiation
- `metric_gate(f1)` — check F1 ≥ 0.80
- `champion_challenger_gate(new_f1, endpoint_name)` — compare against production model
- `register_model(ml_client, job_name, ...)` — model registration logic
- `deploy_model(ml_client, model_name, ...)` — endpoint deployment logic
- `resolve_data_assets(ml_client)` — get latest data asset versions

#### 1c. create-pipeline
**Create `pipeline/run_pipeline.py` — combined AML Pipeline Job**

- Define a `train_component` using `command()` with typed inputs (train_data, validation_data, n_estimators, learning_rate, num_leaves) and outputs (model_output as uri_folder)
- Submit training as a Pipeline Job to AML compute
- Resolve latest data asset versions dynamically
- Get or create compute cluster
- After training completes, read metrics from MLflow (F1_weighted, accuracy, AUC)
- **Metric gate**: if F1_weighted < 0.80, log warning and exit without registering
- Register model in AML Model Registry
- **Champion/challenger gate**: compare new model F1 against current production F1. If new ≤ current, log "current model is better" and exit without deploying
- Deploy to Managed Online Endpoint
- Print AML Studio URL for the pipeline run

#### 1d. update-docs-phase1
**Document the automated pipeline**

Update `docs/implementation/20260410-ml-pipeline-spike.md` with:
- Combined Pipeline Job (how `run_pipeline.py` works end-to-end)
- Metric gates (minimum F1 threshold + champion/challenger comparison)
- Updated pipeline scripts table
- Updated flow diagram showing gates
- Decision log (why combined, why gates, why step 02 stays local)

---

### Phase 2 — Hyperparameter Sweep
*AML Sweep Job to automatically find optimal hyperparameters*

#### 2a. create-sweep
**Create `pipeline/run_sweep.py` — AML Sweep Job**

- Reuse the same `train_component` definition from phase 1
- Define search space:
  - `n_estimators`: Choice([100, 200, 300, 500])
  - `learning_rate`: Uniform(0.01, 0.1)
  - `num_leaves`: Choice([15, 31, 63, 127])
- Configure sweep:
  - Primary metric: `f1_weighted` (maximize)
  - Sampling: Random (good balance of exploration vs cost)
  - Max total trials: 20
  - Max concurrent trials: 4 (limited by cluster size)
  - Early termination: BanditPolicy (kills poor runs early to save compute)
- After sweep completes, get best trial's metrics and hyperparameters
- **Same metric + champion/challenger gates** as run_pipeline.py (via _helpers.py)
- Register + deploy best trial's model (if gates pass)
- Print AML Studio URL for the sweep visualization

#### 2b. update-docs-phase2
**Document the sweep job**

Add to implementation docs:
- Sweep Job (how to run, search space, how to interpret results)
- When to use `run_pipeline.py` vs `run_sweep.py`

---

### Phase 3 — Feature Importance Logging
*Extract and log LightGBM feature importance for explainability*

#### 3a. feature-importance
**Add feature importance logging to `src/train.py`**

- After training, extract `model.feature_importance_` with feature names
- Save feature importance as `{model_output}/feature_importance.json`
- Print sorted feature importance table to stdout
- Log feature importance as MLflow artifact

#### 3b. update-docs-phase3
**Document feature importance**

Add to implementation docs:
- Feature importance logging (what it produces, how to read it)
- How to interpret the output for stakeholder conversations

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| MLflow protobuf conflict in curated env | Try newer curated env first; fall back to custom env definition with pinned protobuf |
| Private `_code` API in deploy script | Out of scope — deploy logic reused as-is from 05-deploy_endpoint.py |
| Sweep cost (20 trials × compute) | BanditPolicy terminates bad runs early; max_concurrent=4 limits parallel spend |
| Pipeline component code upload (SAS restriction) | Use same OAuth upload pattern from 05-deploy_endpoint.py if needed |
| Champion/challenger gate on first run (no production model) | Handle gracefully — if no endpoint exists yet, skip the gate and deploy directly |

---

## Execution order

```
Phase 1: mlflow-logging → extract-shared-helpers → create-pipeline → update-docs-phase1
Phase 2: create-sweep → update-docs-phase2
Phase 3: feature-importance → update-docs-phase3
```

