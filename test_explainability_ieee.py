"""
Simple test script for FeatureCraft explainability feature
using the IEEE Fraud Detection dataset.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from featurecraft import AutoFeatureEngineer, FeatureCraftConfig


def main():
    print("=" * 80)
    print("FeatureCraft Explainability Test - IEEE Fraud Detection")
    print("=" * 80)
    
    # Load IEEE fraud detection data
    print("\n[LOADING] Loading IEEE fraud detection data...")
    df_txn = pd.read_csv("data/ieee-fraud-detection/train_transaction.csv")
    df_identity = pd.read_csv("data/ieee-fraud-detection/train_identity.csv")

    # Merge datasets
    df = df_txn.merge(df_identity, on='TransactionID', how='left')
    print(f"   [OK] Loaded {len(df):,} transactions with {len(df.columns)} columns")

    # Use a sample for quick testing
    df_sample = df.sample(n=min(10000, len(df)), random_state=42)
    print(f"   [OK] Using {len(df_sample):,} samples for testing")

    # Prepare data
    X = df_sample.drop(columns=['isFraud', 'TransactionID'])
    y = df_sample['isFraud']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"   [OK] Train: {len(X_train):,} samples, Test: {len(X_test):,} samples")
    print(f"   [OK] Features: {X_train.shape[1]}")
    
    # Configure basic settings
    config = FeatureCraftConfig(
        use_target_encoding=True,          # Use target encoding
        low_cardinality_max=10,            # Threshold for one-hot encoding
        mid_cardinality_max=100,           # Threshold for target encoding
        scaler_linear="standard",          # Standard scaling for linear models
        verbosity=2,                       # Increase verbosity to see more details
    )
    
    # Create and fit pipeline
    afe = AutoFeatureEngineer(config=config)
    
    print("\n[PROCESSING] Fitting FeatureCraft pipeline with tree-based estimator...")
    print("    (Explanations will show below)\n")

    afe.fit(X_train, y_train, estimator_family="tree")

    # Transform test data
    X_test_transformed = afe.transform(X_test)

    print("\n[SUCCESS] Pipeline completed successfully!")
    print(f"   [OK] Output features: {X_test_transformed.shape[1]}")

    # Save explanations
    print("\n[SAVING] Saving explanations...")
    afe.save_explanation("artifacts/ieee_fraud_explanation.md", format="markdown")
    afe.save_explanation("artifacts/ieee_fraud_explanation.json", format="json")
    print("   [OK] Saved to artifacts/ieee_fraud_explanation.md")
    print("   [OK] Saved to artifacts/ieee_fraud_explanation.json")

    print("\n" + "=" * 80)
    print("[COMPLETE] Test completed! Check the output above to see what transformations were applied.")
    print("=" * 80)


if __name__ == "__main__":
    main()

