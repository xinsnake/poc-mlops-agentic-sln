"""
generate_data.py

Generates ~800 rows of synthetic forecast anomaly records and inserts them
into the Snowflake anomaly_records table.

This mimics what real data from Blue Yonder + Snowflake would look like in
production. The schema and patterns are intentionally realistic:
- ~70% temporary anomalies, ~30% baseline shifts (real-world class imbalance)
- week_id spans 1–52 (used for time-based train/validation split later)
- Feature values are correlated with the label but with realistic noise and
  overlap — some temporary anomalies last longer, some baseline shifts recover
  quickly. This produces probabilities in the 0.3–0.8 range rather than 0/1.

Run this once to populate Snowflake before running data_pipeline.py.
"""

import os
import random
import numpy as np
import pandas as pd
import snowflake.connector
from dotenv import load_dotenv

load_dotenv()

RANDOM_SEED = 42
NUM_ROWS = 800
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DC_REGIONS = ["MEL", "SYD", "BNE", "PER", "ADL"]
LIFECYCLE_STAGES = ["G", "M", "D"]  # Growth, Maintain, Decline

# Noise probability: chance a sample gets features from the "wrong" distribution
# This creates realistic overlap between classes
NOISE_RATE = 0.15


def generate_row(week_id: int) -> dict:
    """Generate a single synthetic anomaly record with realistic noise."""

    anomaly_type = "temporary" if random.random() < 0.70 else "baseline_shift"

    # With NOISE_RATE probability, generate features from the opposite distribution
    # This simulates real-world ambiguity: a short-lived baseline shift, a prolonged
    # temporary spike caused by an unusual event, etc.
    use_opposite_features = random.random() < NOISE_RATE

    if (anomaly_type == "temporary") != use_opposite_features:
        # Temporary anomaly features: short spike, recovers quickly
        weeks_affected = int(np.clip(np.random.normal(2, 1.2), 1, 6))     # mean 2, can bleed into shift range
        forecast_bias = round(random.uniform(-0.6, 0.1), 3)
        forecast_accuracy = round(np.clip(np.random.normal(0.62, 0.12), 0.3, 0.9), 3)
        pct_error = round(np.clip(np.random.normal(0.38, 0.12), 0.1, 0.7), 3)
        volume_of_error = round(np.clip(np.random.normal(200, 100), 30, 600), 1)
    else:
        # Baseline shift features: sustained change, higher error, longer duration
        weeks_affected = int(np.clip(np.random.normal(6, 1.8), 2, 12))    # mean 6, can overlap with temporary
        forecast_bias = round(random.uniform(-0.1, 0.65), 3)
        forecast_accuracy = round(np.clip(np.random.normal(0.48, 0.12), 0.2, 0.78), 3)
        pct_error = round(np.clip(np.random.normal(0.52, 0.13), 0.2, 0.8), 3)
        volume_of_error = round(np.clip(np.random.normal(480, 150), 100, 900), 1)

    return {
        "week_id": week_id,
        "sku_id": f"SKU-{random.randint(1000, 9999)}",
        "store_id": f"STORE-{random.randint(1, 450):03d}",
        "dc_region": random.choice(DC_REGIONS),
        "product_lifecycle": random.choice(LIFECYCLE_STAGES),
        "is_seasonal": random.random() < 0.3,
        "forecast_bias": forecast_bias,
        "forecast_accuracy": forecast_accuracy,
        "volume_of_error": volume_of_error,
        "pct_error": pct_error,
        "weeks_affected": weeks_affected,
        "prior_anomaly_count": random.randint(0, 15),
        "anomaly_type": anomaly_type,
    }


def generate_dataset(num_rows: int) -> pd.DataFrame:
    """Generate the full dataset spread across 52 weeks."""
    rows = []
    for i in range(num_rows):
        # Distribute rows across 52 weeks (not perfectly even — mimics reality)
        week_id = random.randint(1, 52)
        rows.append(generate_row(week_id))

    df = pd.DataFrame(rows)
    print(f"Generated {len(df)} rows")
    print(f"Label distribution:\n{df['anomaly_type'].value_counts()}")
    print(f"Week range: {df['week_id'].min()} – {df['week_id'].max()}")
    return df


def insert_into_snowflake(df: pd.DataFrame) -> None:
    """Connect to Snowflake and insert all rows into anomaly_records."""
    conn = snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
    )

    insert_sql = """
        INSERT INTO anomaly_records (
            week_id, sku_id, store_id, dc_region, product_lifecycle,
            is_seasonal, forecast_bias, forecast_accuracy, volume_of_error,
            pct_error, weeks_affected, prior_anomaly_count, anomaly_type
        ) VALUES (
            %(week_id)s, %(sku_id)s, %(store_id)s, %(dc_region)s,
            %(product_lifecycle)s, %(is_seasonal)s, %(forecast_bias)s,
            %(forecast_accuracy)s, %(volume_of_error)s, %(pct_error)s,
            %(weeks_affected)s, %(prior_anomaly_count)s, %(anomaly_type)s
        )
    """

    try:
        cursor = conn.cursor()
        rows = df.to_dict(orient="records")
        cursor.executemany(insert_sql, rows)
        conn.commit()
        print(f"Inserted {len(rows)} rows into Snowflake anomaly_records table")
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    df = generate_dataset(NUM_ROWS)
    insert_into_snowflake(df)
    print("Done.")
