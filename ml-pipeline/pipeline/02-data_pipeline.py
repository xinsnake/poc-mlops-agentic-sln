"""
data_pipeline.py

Extracts anomaly records from Snowflake, processes them, performs a
time-based 60/40 train/validation split, and registers both splits
as Data Assets in Azure ML.

Run this after generate_data.py has populated Snowflake.
Run this before submit_training.py.
"""

import os
import pandas as pd
from dotenv import load_dotenv
import snowflake.connector
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

load_dotenv()

# Paths for the split datasets saved locally before uploading to AML
TRAIN_PATH = "data/train.csv"
VALIDATION_PATH = "data/validation.csv"

# week_id cutoff: weeks 1-31 = training (60%), weeks 32-52 = validation (40%)
SPLIT_WEEK = 31


def extract_from_snowflake() -> pd.DataFrame:
    """Pull all anomaly records from Snowflake."""
    print("Connecting to Snowflake...")
    conn = snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
    )

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM anomaly_records ORDER BY week_id")
        columns = [col[0].lower() for col in cursor.description]
        rows = cursor.fetchall()
        df = pd.DataFrame(rows, columns=columns)
    finally:
        cursor.close()
        conn.close()

    print(f"Extracted {len(df)} rows from Snowflake")
    return df


def process(df: pd.DataFrame) -> pd.DataFrame:
    """
    Light processing:
    - Encode categorical columns as integers
    - Normalise numeric columns to 0-1 range
    - Handle nulls
    """
    print("Processing data...")

    df = df.dropna()

    dc_region_map = {r: i for i, r in enumerate(sorted(df["dc_region"].unique()))}
    lifecycle_map = {"G": 0, "M": 1, "D": 2}

    df["dc_region_encoded"] = df["dc_region"].map(dc_region_map)
    df["product_lifecycle_encoded"] = df["product_lifecycle"].map(lifecycle_map)
    df["is_seasonal_encoded"] = df["is_seasonal"].astype(int)

    # label: 0 = temporary, 1 = baseline_shift
    df["label"] = (df["anomaly_type"] == "baseline_shift").astype(int)

    numeric_cols = ["forecast_bias", "forecast_accuracy", "volume_of_error", "pct_error"]
    for col in numeric_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max > col_min:
            df[col] = (df[col] - col_min) / (col_max - col_min)
        else:
            df[col] = 0.5  # constant column — set to midpoint

    print(f"After processing: {len(df)} rows, {len(df.columns)} columns")
    return df


def split(df: pd.DataFrame) -> tuple:
    """
    Time-based split:
    - Training:   week_id <= SPLIT_WEEK  (~60%)
    - Validation: week_id >  SPLIT_WEEK  (~40%)
    """
    train = df[df["week_id"] <= SPLIT_WEEK].copy()
    validation = df[df["week_id"] > SPLIT_WEEK].copy()

    print(f"Training set:   {len(train)} rows (weeks 1-{SPLIT_WEEK})")
    print(f"Validation set: {len(validation)} rows (weeks {SPLIT_WEEK + 1}-52)")
    print(f"Label distribution (train):\n{train['anomaly_type'].value_counts()}")
    print(f"Label distribution (validation):\n{validation['anomaly_type'].value_counts()}")

    return train, validation


def save_locally(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    """Save splits to local CSV files before uploading to AML."""
    os.makedirs("data", exist_ok=True)
    train.to_csv(TRAIN_PATH, index=False)
    validation.to_csv(VALIDATION_PATH, index=False)
    print(f"Saved training set to {TRAIN_PATH}")
    print(f"Saved validation set to {VALIDATION_PATH}")


def register_aml_data_assets() -> None:
    """
    Upload CSVs to AML's default blob storage using identity-based auth
    (OAuth via DefaultAzureCredential / az login), then register as versioned Data Assets.

    This avoids storage account key access, which may be blocked by Azure Policy.
    """
    print("Connecting to Azure ML...")
    credential = DefaultAzureCredential()

    ml_client = MLClient(
        credential=credential,
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_ML_WORKSPACE"],
    )

    # Get the default datastore's storage account and container
    datastore = ml_client.datastores.get_default()
    storage_account = datastore.account_name
    container = datastore.container_name
    blob_base_url = f"https://{storage_account}.blob.core.windows.net"

    print(f"Uploading to blob storage: {storage_account}/{container}")
    blob_service = BlobServiceClient(account_url=blob_base_url, credential=credential)
    container_client = blob_service.get_container_client(container)

    datasets = [
        (TRAIN_PATH, "spike-data/train.csv", "anomaly-train",
         "Training split (weeks 1-31) of synthetic forecast anomaly records"),
        (VALIDATION_PATH, "spike-data/validation.csv", "anomaly-validation",
         "Validation split (weeks 32-52) of synthetic forecast anomaly records"),
    ]

    sub = os.environ["AZURE_SUBSCRIPTION_ID"]
    rg = os.environ["AZURE_RESOURCE_GROUP"]
    ws = os.environ["AZURE_ML_WORKSPACE"]

    for local_path, blob_name, asset_name, description in datasets:
        with open(local_path, "rb") as f:
            container_client.upload_blob(name=blob_name, data=f, overwrite=True)
        print(f"Uploaded {local_path} to blob: {blob_name}")

        blob_uri = (
            f"azureml://subscriptions/{sub}"
            f"/resourcegroups/{rg}"
            f"/workspaces/{ws}"
            f"/datastores/{datastore.name}/paths/{blob_name}"
        )

        asset = Data(
            name=asset_name,
            path=blob_uri,
            type=AssetTypes.URI_FILE,
            description=description,
        )
        ml_client.data.create_or_update(asset)
        print(f"Registered AML Data Asset: {asset_name}")


if __name__ == "__main__":
    df = extract_from_snowflake()
    df = process(df)
    train, validation = split(df)
    save_locally(train, validation)
    register_aml_data_assets()
    print("Done.")
