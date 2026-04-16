"""
run_pipeline.py

Combined AML Pipeline Job: train → metric gate → register → champion/challenger gate → deploy.

This runs locally — it submits a Pipeline Job to AML, waits for completion,
then applies metric gates to decide whether to register and deploy the model.

Usage:
    cd ml-pipeline
    python pipeline/run_pipeline.py

Flow:
    1. Resolve latest data assets (anomaly-train, anomaly-validation)
    2. Get/create compute cluster
    3. Submit training Pipeline Job to AML (train.py on compute)
    4. METRIC GATE: F1_weighted >= 0.80? If no → stop
    5. Register model in AML Model Registry
    6. CHAMPION/CHALLENGER GATE: new F1 > production F1? If no → stop
    7. Deploy to Managed Online Endpoint
"""

import os
import sys
import mlflow
from dotenv import load_dotenv
from azure.ai.ml import command, Input, Output, dsl
from azure.ai.ml.constants import AssetTypes, InputOutputModes

from _helpers import (
    get_ml_client,
    get_or_create_compute,
    resolve_data_assets,
    metric_gate,
    champion_challenger_gate,
    register_model,
    deploy_model,
    upload_and_register_code,
    get_aml_studio_run_url,
)

load_dotenv()


def make_train_component():
    """Define the training component — reusable by run_sweep.py."""
    src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
    return command(
        display_name="anomaly-classifier-training",
        description="LightGBM training job for forecast anomaly classification (UC4 spike)",
        code=src_dir,
        command=(
            "python train.py"
            " --train-data ${{inputs.train_data}}"
            " --validation-data ${{inputs.validation_data}}"
            " --model-output ${{outputs.model_output}}"
            " --n-estimators ${{inputs.n_estimators}}"
            " --learning-rate ${{inputs.learning_rate}}"
            " --num-leaves ${{inputs.num_leaves}}"
        ),
        inputs={
            "train_data": Input(type=AssetTypes.URI_FILE),
            "validation_data": Input(type=AssetTypes.URI_FILE),
            "n_estimators": Input(type="integer", default=200),
            "learning_rate": Input(type="number", default=0.05),
            "num_leaves": Input(type="integer", default=31),
        },
        outputs={
            "model_output": Output(type=AssetTypes.URI_FOLDER),
        },
        environment=os.environ["AML_ENVIRONMENT"],
    )


def get_job_metrics(ml_client, job_name: str) -> dict:
    """
    Read metrics from a completed AML job via the MLflow tracking API.
    Handles both pipeline jobs (navigates to child job) and standalone command jobs.
    """
    # Resolve the training child job for pipeline jobs
    job = ml_client.jobs.get(job_name)
    if hasattr(job, "jobs") and job.jobs:
        child_jobs = list(ml_client.jobs.list(parent_job_name=job_name))
        if not child_jobs:
            raise RuntimeError(f"Pipeline job {job_name} has no child jobs")
        run_id = child_jobs[0].name
    else:
        run_id = job_name

    # Read metrics via MLflow tracking API
    tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    mlflow_client = mlflow.tracking.MlflowClient()

    run = mlflow_client.get_run(run_id)
    metrics = run.data.metrics

    if not metrics:
        raise RuntimeError(f"No metrics found for job {run_id}. Check train.py logged metrics via MLflow.")

    print(f"\nTraining metrics (job: {run_id}):")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")
    return metrics


def get_training_job_name(ml_client, pipeline_job_name: str) -> str:
    """Get the training child job name from a pipeline job (for model registration)."""
    job = ml_client.jobs.get(pipeline_job_name)
    if hasattr(job, "jobs") and job.jobs:
        child_jobs = list(ml_client.jobs.list(parent_job_name=pipeline_job_name))
        if not child_jobs:
            raise RuntimeError(f"Pipeline job {pipeline_job_name} has no child jobs")
        return child_jobs[0].name
    return pipeline_job_name


def run():
    ml_client = get_ml_client()

    # ── Step 1: Resolve data + compute ───────────────────────────────────────
    train_version, val_version = resolve_data_assets(ml_client)
    compute_name = get_or_create_compute(ml_client)

    # ── Step 2: Build and submit pipeline ────────────────────────────────────
    train_component = make_train_component()

    @dsl.pipeline(
        display_name="anomaly-classifier-pipeline",
        description="Train → gate → register → deploy pipeline for UC4 anomaly classification",
        compute=compute_name,
    )
    def training_pipeline(train_data, validation_data):
        train_step = train_component(
            train_data=train_data,
            validation_data=validation_data,
        )
        return {"model_output": train_step.outputs.model_output}

    pipeline_job = training_pipeline(
        train_data=Input(
            type=AssetTypes.URI_FILE,
            path=f"azureml:anomaly-train:{train_version}",
            mode=InputOutputModes.DOWNLOAD,
        ),
        validation_data=Input(
            type=AssetTypes.URI_FILE,
            path=f"azureml:anomaly-validation:{val_version}",
            mode=InputOutputModes.DOWNLOAD,
        ),
    )

    submitted = ml_client.jobs.create_or_update(pipeline_job)
    print(f"\nPipeline submitted: {submitted.display_name}")
    print(f"Job name: {submitted.name}")
    print(f"View in AML Studio: {get_aml_studio_run_url(submitted.name)}")
    print(f"\nStreaming logs (Ctrl+C to stop following, job continues running)...")

    ml_client.jobs.stream(submitted.name)

    # ── Step 3: Read metrics ─────────────────────────────────────────────────
    metrics = get_job_metrics(ml_client, submitted.name)
    f1 = metrics.get("f1_weighted")
    if f1 is None:
        print("\n❌ Could not read f1_weighted metric from training job. Aborting.")
        sys.exit(1)

    # ── Step 4: Metric gate ──────────────────────────────────────────────────
    if not metric_gate(f1):
        sys.exit(1)

    # ── Step 5: Register model ───────────────────────────────────────────────
    training_job_name = get_training_job_name(ml_client, submitted.name)
    model_version = register_model(ml_client, training_job_name, f1)

    # ── Step 6: Champion/challenger gate ─────────────────────────────────────
    if not champion_challenger_gate(ml_client, f1):
        print("\nModel registered but NOT deployed (current production model is better).")
        sys.exit(0)

    # ── Step 7: Deploy ───────────────────────────────────────────────────────
    deploy_model(ml_client)
    print("\n✅ Pipeline complete: trained → registered → deployed")


if __name__ == "__main__":
    run()
