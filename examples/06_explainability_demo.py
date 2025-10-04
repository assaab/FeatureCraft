"""
Explainability Demo - Understanding What FeatureCraft is Doing

This example demonstrates the new explainability feature that shows users
what transformations are being applied and WHY.
"""

from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

from featurecraft import AutoFeatureEngineer, FeatureCraftConfig


def demo_basic_explainability():
    """Demo: Basic explainability with auto-printing."""
    print("=" * 80)
    print("DEMO 1: Basic Explainability (Auto-Print)")
    print("=" * 80)
    
    # Load dataset
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    X = df.drop(columns=["target"])
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create pipeline with explainability enabled (default)
    config = FeatureCraftConfig(
        explain_transformations=True,  # Enable explanations (default)
        explain_auto_print=True,       # Auto-print after fit (default)
    )
    
    afe = AutoFeatureEngineer(config=config)
    
    # Fit - explanations will be automatically printed
    print("\n🔧 Fitting pipeline...\n")
    afe.fit(X_train, y_train, estimator_family="tree")
    
    print("\n✅ Pipeline fitted successfully!")


def demo_manual_explanation():
    """Demo: Manual explanation access and export."""
    print("\n\n" + "=" * 80)
    print("DEMO 2: Manual Explanation Access")
    print("=" * 80)
    
    # Load dataset
    data = load_iris(as_frame=True)
    df = data.frame
    X = df.drop(columns=["target"])
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Disable auto-printing
    config = FeatureCraftConfig(
        explain_transformations=True,
        explain_auto_print=False,  # Don't auto-print
        skew_threshold=0.5,        # Lower threshold to trigger power transform
    )
    
    afe = AutoFeatureEngineer(config=config)
    afe.fit(X_train, y_train, estimator_family="linear")
    
    # Access explanation programmatically
    explanation = afe.get_explanation()
    
    # Print to console when you want
    print("\n📊 Printing explanation on demand:\n")
    afe.print_explanation()
    
    # Export to files
    print("\n💾 Exporting explanations...")
    afe.save_explanation("artifacts/iris_explanation.md", format="markdown")
    afe.save_explanation("artifacts/iris_explanation.json", format="json")
    print("   ✓ Saved to artifacts/iris_explanation.md")
    print("   ✓ Saved to artifacts/iris_explanation.json")
    
    # Access explanation data programmatically
    print("\n📈 Programmatic access:")
    print(f"   • Total explanations: {len(explanation.explanations)}")
    print(f"   • Task type: {explanation.task_type}")
    print(f"   • Estimator family: {explanation.estimator_family}")
    print(f"   • Features: {explanation.n_features_in} → {explanation.n_features_out}")
    
    # Get markdown string
    md_content = explanation.to_markdown()
    print(f"   • Markdown length: {len(md_content)} characters")


def demo_complex_pipeline():
    """Demo: Complex pipeline with many transformations."""
    print("\n\n" + "=" * 80)
    print("DEMO 3: Complex Pipeline with Many Decisions")
    print("=" * 80)
    
    # Create synthetic dataset with various column types
    import numpy as np
    
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        # Numeric features
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.exponential(50000, n_samples),  # Skewed
        'score': np.random.normal(100, 15, n_samples),
        'rating': np.random.uniform(1, 5, n_samples),
        
        # Low cardinality categorical
        'gender': np.random.choice(['M', 'F', 'Other'], n_samples),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        
        # Medium cardinality categorical
        'city': np.random.choice([f'City_{i}' for i in range(30)], n_samples),
        'occupation': np.random.choice([f'Job_{i}' for i in range(40)], n_samples),
        
        # High cardinality categorical
        'user_id': [f'user_{i}' for i in np.random.randint(0, 500, n_samples)],
        
        # Target
        'target': np.random.randint(0, 2, n_samples),
    })
    
    # Add some missing values
    df.loc[np.random.choice(df.index, 50), 'income'] = np.nan
    df.loc[np.random.choice(df.index, 30), 'city'] = np.nan
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Complex configuration
    config = FeatureCraftConfig(
        explain_transformations=True,
        explain_auto_print=True,
        use_target_encoding=True,
        use_leave_one_out_te=False,  # Use out-of-fold TE
        cv_n_splits=5,
        skew_threshold=0.75,
        scaler_linear="standard",
        scaler_robust_if_outliers=True,
        outlier_share_threshold=0.05,
        winsorize=True,
        clip_percentiles=(0.01, 0.99),
        low_cardinality_max=5,
        mid_cardinality_max=50,
    )
    
    afe = AutoFeatureEngineer(config=config)
    
    print("\n🔧 Fitting complex pipeline...\n")
    afe.fit(X_train, y_train, estimator_family="linear")
    
    print("\n✅ Complex pipeline fitted successfully!")
    
    # Export with pipeline artifacts
    afe.export("artifacts/complex_pipeline")
    print("\n💾 Pipeline and explanation exported to artifacts/complex_pipeline/")


def demo_disable_explanations():
    """Demo: Disable explanations for performance."""
    print("\n\n" + "=" * 80)
    print("DEMO 4: Disabling Explanations")
    print("=" * 80)
    
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    X = df.drop(columns=["target"])
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Disable explainability for production/performance
    config = FeatureCraftConfig(
        explain_transformations=False,  # Disable explanations
    )
    
    afe = AutoFeatureEngineer(config=config)
    
    print("\n🔧 Fitting pipeline (no explanations)...\n")
    afe.fit(X_train, y_train, estimator_family="tree")
    
    print("✅ Pipeline fitted (silent mode - no explanations printed)")
    
    # Trying to access explanations will raise an error
    try:
        afe.get_explanation()
    except RuntimeError as e:
        print(f"\n⚠️  Expected error: {e}")


if __name__ == "__main__":
    # Run all demos
    demo_basic_explainability()
    demo_manual_explanation()
    demo_complex_pipeline()
    demo_disable_explanations()
    
    print("\n\n" + "=" * 80)
    print("🎉 All explainability demos completed!")
    print("=" * 80)

