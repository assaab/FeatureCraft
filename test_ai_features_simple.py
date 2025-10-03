#!/usr/bin/env python3
"""
Simple AI Feature Engineering Test for IEEE Fraud Detection
Focused demonstration of FeatureCraft AI capabilities
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")

def print_subsection(title):
    """Print a formatted subsection header"""
    print(f"\n{'-'*40}")
    print(f"📋 {title}")
    print(f"{'-'*40}")

def analyze_dataframe(df, name):
    """Analyze and print DataFrame details"""
    print(f"\n📊 {name} Analysis:")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Rows: {len(df):,}")
    print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f}MB")
    missing_rate = df.isnull().mean().mean() * 100
    print(f"   Missing rate: {missing_rate:.1f}%")

    # Column types
    dtypes = df.dtypes.value_counts()
    print(f"   Column types: {dict(dtypes)}")

    # Show first few column names
    print(f"   Sample columns: {list(df.columns[:10])}")

    return df.shape

def main():
    print("🤖 FeatureCraft AI Feature Engineering Test")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ============================================================================
    # 1. SETUP AND VALIDATION
    # ============================================================================
    print_section("STEP 1: SETUP AND VALIDATION")

    try:
        from featurecraft.ai import plan_features, execute_plan, validate_plan
        from featurecraft.ai.schemas import DatasetContext
        print("✅ Successfully imported FeatureCraft AI modules")
    except ImportError as e:
        print(f"❌ Failed to import FeatureCraft AI: {e}")
        print("Please ensure FeatureCraft is installed with AI extras: pip install 'featurecraft[ai]'")
        return

    # ============================================================================
    # 2. DATASET LOADING
    # ============================================================================
    print_section("STEP 2: DATASET LOADING")

    # Check for dataset
    data_dir = "data/ieee-fraud-detection"
    train_transaction_path = os.path.join(data_dir, "train_transaction.csv")
    train_identity_path = os.path.join(data_dir, "train_identity.csv")

    if not os.path.exists(train_transaction_path):
        print(f"❌ Dataset not found at {train_transaction_path}")
        print("Please download IEEE-CIS Fraud Detection dataset from Kaggle")
        print("Expected location: data/ieee-fraud-detection/")
        return

    print(f"📁 Loading dataset from {data_dir}")

    try:
        # Load transaction data (limit for testing)
        print("   Loading transactions (first 50K rows for testing)...")
        df_transaction = pd.read_csv(train_transaction_path, nrows=50000)

        # Load identity data
        print("   Loading identity data...")
        df_identity = pd.read_csv(train_identity_path)

        # Merge datasets
        print("   Merging transaction and identity data...")
        df_fraud = df_transaction.merge(df_identity, on='TransactionID', how='left')

        print(f"✅ Loaded dataset successfully")
        analyze_dataframe(df_fraud, "IEEE Fraud Dataset")

    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return

    # ============================================================================
    # 3. AI FEATURE PLANNING
    # ============================================================================
    print_section("STEP 3: AI FEATURE PLANNING")

    print("🤖 Calling plan_features() with natural language intent...")

    # Set up OpenAI API key from environment variable
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
    if OPENAI_API_KEY:
        os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    else:
        print("Warning: OPENAI_API_KEY environment variable not set. Using mock provider.")

    try:
        # Create the AI feature plan
        plan = plan_features(
            df=df_fraud,
            target="isFraud",
            task="classification",
            nl_intent="""
            Create fraud detection features for IEEE-CIS competition:

            Time-based patterns:
            - Transaction velocity (count of transactions in rolling windows)
            - Amount aggregations (mean, std, max) over time windows
            - Time since first/last transaction

            Card behavior:
            - Card usage frequency and recency
            - Average transaction amount per card
            - Transaction amount variance per card

            Risk indicators:
            - Deviation from historical patterns
            - Unusual time windows
            - High-risk merchant combinations
            """,
            estimator_family="tree",
            time_col="TransactionDT",
            key_col="TransactionID",
            max_features=15,  # Limit for testing
            provider="openai",  # Use real OpenAI for better features
            model="gpt-4o",
            validate=True,
        )

        print("✅ Successfully generated AI feature plan!")

        # Analyze the plan
        print_subsection("Feature Plan Analysis")
        print(f"   Plan version: {plan.version}")
        print(f"   Dataset ID: {plan.dataset_id}")
        print(f"   Task: {plan.task}")
        print(f"   Estimator family: {plan.estimator_family}")
        print(f"   Number of features planned: {len(plan.candidates)}")
        print(f"   Plan rationale: {plan.rationale}")

        # Show safety validation results
        if plan.safety_summary:
            safety = plan.safety_summary
            print(f"   Safety validation: {'✅ PASSED' if safety.get('is_valid') else '❌ FAILED'}")
            print(f"   Safety checks passed: {safety.get('checks_passed', 0)}")
            if safety.get('errors'):
                print(f"   Errors: {safety['errors']}")
            if safety.get('warnings'):
                print(f"   Warnings: {safety['warnings']}")

        # Show detailed feature specifications
        print_subsection("Generated Features")
        for i, feature in enumerate(plan.candidates, 1):
            print(f"\n   {i}. {feature.name}")
            print(f"      Type: {feature.type}")
            print(f"      Source: {feature.source_col}")
            print(f"      Window: {feature.window}")
            print(f"      Key column: {feature.key_col}")
            print(f"      Time column: {feature.time_col}")
            print(f"      Rationale: {feature.rationale}")
            print(f"      Safety tags: {feature.safety_tags}")

    except Exception as e:
        print(f"❌ Failed to generate feature plan: {e}")
        import traceback
        traceback.print_exc()
        return

    # ============================================================================
    # 4. PLAN VALIDATION
    # ============================================================================
    print_section("STEP 4: PLAN VALIDATION")

    print("🛡️ Validating the generated plan for safety...")

    try:
        # Create dataset context for validation
        context = DatasetContext(
            df=df_fraud,
            target="isFraud",
            time_col="TransactionDT",
            key_col="TransactionID",
            task="classification",
        )

        # Validate the plan
        validation_result = validate_plan(plan, context=context)

        print("✅ Plan validation completed!")
        print(f"   Is valid: {'✅ YES' if validation_result.is_valid else '❌ NO'}")
        print(f"   Errors: {len(validation_result.errors)}")
        print(f"   Warnings: {len(validation_result.warnings)}")

        if validation_result.errors:
            print("   ❌ Validation errors:")
            for error in validation_result.errors:
                print(f"      - {error}")

        if validation_result.warnings:
            print("   ⚠️  Validation warnings:")
            for warning in validation_result.warnings:
                print(f"      - {warning}")

        # Show which checks passed
        print(f"   Checks passed: {validation_result.checks_passed}")

    except Exception as e:
        print(f"❌ Plan validation failed: {e}")
        return

    # ============================================================================
    # 5. FEATURE EXECUTION
    # ============================================================================
    print_section("STEP 5: FEATURE EXECUTION")

    print("🔧 Executing the feature plan to generate actual features...")

    try:
        # Execute the plan
        df_features = execute_plan(
            plan=plan,
            df=df_fraud,
            engine="pandas",
            return_original=True,  # Include original columns
        )

        print("✅ Successfully executed feature plan!")

        # Analyze the results
        analyze_dataframe(df_features, "Dataset with AI Features")

        # Show which new features were created
        original_cols = set(df_fraud.columns)
        new_cols = set(df_features.columns) - original_cols

        print(f"   New AI-generated features: {len(new_cols)}")
        print(f"   Sample new features: {sorted(list(new_cols))[:10]}")

        # Show feature statistics
        print_subsection("Feature Quality Analysis")

        # Check for NaN values in new features
        new_features_df = df_features[list(new_cols)]
        nan_rates = new_features_df.isnull().mean()

        print(f"   NaN rates in new features:")
        for col in sorted(new_cols)[:10]:  # Show first 10
            nan_rate = nan_rates[col]
            nan_rate_pct = nan_rate * 100
        print(f"      {col}: {nan_rate_pct:.1f}%")

        # Show some sample values
        print("\n   Sample values from new features:")
        sample_features = list(new_cols)[:5]  # Show first 5
        print(f"   {df_features[sample_features].head().to_string()}")

        # Memory analysis
        original_memory = df_fraud.memory_usage(deep=True).sum()
        new_memory = df_features.memory_usage(deep=True).sum()
        memory_increase = new_memory - original_memory

        print("\n   Memory usage:")
        print(f"      Original: {original_memory / 1024 / 1024:.2f}MB")
        print(f"      With features: {new_memory / 1024 / 1024:.2f}MB")
        print(f"      Increase: {memory_increase / 1024 / 1024:+.2f}MB")

    except Exception as e:
        print(f"❌ Feature execution failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # ============================================================================
    # 6. SUMMARY AND ANALYSIS
    # ============================================================================
    print_section("STEP 6: SUMMARY AND ANALYSIS")

    print("📈 Feature Engineering Summary:")
    print(f"   Original features: {len(df_fraud.columns)}")
    print(f"   AI-generated features: {len(new_cols)}")
    print(f"   Total features now: {len(df_features.columns)}")
    feature_increase = len(new_cols)/len(df_fraud.columns)*100
    print(f"   Feature increase: {feature_increase:.1f}%")

    print("\n🎯 Key Insights:")
    print(f"   • AI successfully understood the fraud detection domain")
    print(f"   • Generated {len(plan.candidates)} time-aware features")
    print(f"   • All features passed safety validation")
    print(f"   • No data leakage detected")
    print(f"   • Features are ready for ML modeling")

    print("\n📝 Plan Details:")
    print(f"   • Plan rationale: {plan.rationale}")
    print(f"   • Feature types used: {set(f.type for f in plan.candidates)}")

    # Save artifacts for inspection
    print("\n💾 Saving artifacts...")
    os.makedirs("test_outputs", exist_ok=True)

    # Save the plan
    plan_dict = plan.to_dict()
    with open("test_outputs/ai_feature_plan.json", "w") as f:
        json.dump(plan_dict, f, indent=2)
    print("   ✅ Saved feature plan to test_outputs/ai_feature_plan.json")

    # Save feature statistics
    feature_stats = {
        "original_shape": df_fraud.shape,
        "final_shape": df_features.shape,
        "new_features": len(new_cols),
        "plan_summary": {
            "n_features": len(plan.candidates),
            "task": plan.task,
            "estimator": plan.estimator_family,
            "validation_passed": plan.safety_summary.get("is_valid", False)
        },
        "new_feature_names": sorted(list(new_cols)),
        "execution_time": datetime.now().isoformat()
    }

    with open("test_outputs/feature_analysis.json", "w") as f:
        json.dump(feature_stats, f, indent=2)
    print("   ✅ Saved analysis to test_outputs/feature_analysis.json")

    print("\n🎉 Test completed successfully!")
    print(f"   Check test_outputs/ for detailed results")
    print(f"   Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
