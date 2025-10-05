"""
@08_feature_interactions_demo.py

Comprehensive demonstration of Feature Interactions in FeatureCraft.

This example showcases all 6 types of feature interactions:
1. Arithmetic (add, subtract, multiply, divide)
2. Polynomial (x², x³, cross-products)
3. Ratios & Proportions
4. Multi-way Products
5. Categorical × Numeric
6. Binned Interactions
"""

import warnings
import sys
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score

# Add the src directory to the path so we can import local featurecraft modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

warnings.filterwarnings('ignore')


def create_synthetic_dataset_with_interactions():
    """Create a synthetic dataset with known interaction effects."""
    np.random.seed(42)
    n_samples = 1000
    
    # Base features
    age = np.random.randint(18, 80, n_samples)
    income = np.random.randint(20000, 150000, n_samples)
    credit_score = np.random.randint(300, 850, n_samples)
    debt = np.random.randint(0, 100000, n_samples)
    savings = np.random.randint(0, 200000, n_samples)
    
    # Categorical features
    education = np.random.choice(['high_school', 'bachelors', 'masters', 'phd'], n_samples, 
                                 p=[0.3, 0.4, 0.2, 0.1])
    employment = np.random.choice(['full_time', 'part_time', 'self_employed', 'unemployed'], n_samples,
                                  p=[0.6, 0.2, 0.15, 0.05])
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'credit_score': credit_score,
        'debt': debt,
        'savings': savings,
        'education': education,
        'employment': employment,
    })
    
    # Target with interaction effects
    # Loan approval depends on:
    # 1. debt_to_income ratio (arithmetic)
    # 2. age × income interaction (multiplicative)
    # 3. credit_score² (polynomial)
    # 4. savings_rate (ratio)
    # 5. education level effects on income (categorical × numeric)
    
    debt_to_income = debt / (income + 1)
    savings_rate = savings / (income + 1)
    education_bonus = pd.Series(education).map({
        'high_school': 0, 'bachelors': 1, 'masters': 2, 'phd': 3
    }).values
    
    # Complex target formula with interactions
    score = (
        0.5 * (credit_score / 850) +  # Normalized credit score
        0.3 * (1 - debt_to_income) +  # Lower debt-to-income is better
        0.2 * savings_rate +  # Higher savings rate is better
        0.1 * (age * income) / 1e7 +  # Age-income interaction
        0.15 * education_bonus / 3 +  # Education effect
        -0.1 * (debt_to_income ** 2)  # Non-linear penalty
    )
    
    # Add noise and threshold for binary classification
    noise = np.random.normal(0, 0.15, n_samples)
    score = score + noise
    target = (score > np.median(score)).astype(int)
    
    return df, pd.Series(target, name='loan_approved')


def demo_all_interactions():
    """Demonstrate all feature interaction types."""
    print("=" * 80)
    print("FeatureCraft: Feature Interactions Demo")
    print("=" * 80)
    
    # Create synthetic dataset
    print("\n1. Creating synthetic dataset with known interaction effects...")
    X, y = create_synthetic_dataset_with_interactions()
    print(f"   Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Target distribution: {y.value_counts().to_dict()}")
    print(f"\n   Features: {list(X.columns)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # ========== Baseline: No Interactions ==========
    print("\n" + "=" * 80)
    print("2. BASELINE: Training without feature interactions")
    print("=" * 80)
    
    from featurecraft import AutoFeatureEngineer, FeatureCraftConfig
    
    config_baseline = FeatureCraftConfig(
        interactions_enabled=False,
        explain_transformations=True,
        explain_auto_print=False,
        random_state=42,
    )
    
    fe_baseline = AutoFeatureEngineer(config=config_baseline)
    fe_baseline.fit(X_train, y_train)
    
    X_train_baseline = fe_baseline.transform(X_train)
    X_test_baseline = fe_baseline.transform(X_test)
    
    print(f"\n   Features after preprocessing: {X_train_baseline.shape[1]}")
    
    # Train models
    lr_baseline = LogisticRegression(max_iter=1000, random_state=42)
    lr_baseline.fit(X_train_baseline, y_train)
    
    rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_baseline.fit(X_train_baseline, y_train)
    
    # Evaluate
    lr_acc_baseline = accuracy_score(y_test, lr_baseline.predict(X_test_baseline))
    lr_auc_baseline = roc_auc_score(y_test, lr_baseline.predict_proba(X_test_baseline)[:, 1])
    
    rf_acc_baseline = accuracy_score(y_test, rf_baseline.predict(X_test_baseline))
    rf_auc_baseline = roc_auc_score(y_test, rf_baseline.predict_proba(X_test_baseline)[:, 1])
    
    print(f"\n   Logistic Regression - Accuracy: {lr_acc_baseline:.4f}, AUC: {lr_auc_baseline:.4f}")
    print(f"   Random Forest       - Accuracy: {rf_acc_baseline:.4f}, AUC: {rf_auc_baseline:.4f}")
    
    # ========== With Interactions ==========
    print("\n" + "=" * 80)
    print("3. WITH FEATURE INTERACTIONS: Training with all interaction types")
    print("=" * 80)
    
    config_interactions = FeatureCraftConfig(
        # Enable feature interactions
        interactions_enabled=True,
        
        # Configure interaction types
        interactions_use_arithmetic=True,
        interactions_arithmetic_ops=['add', 'subtract', 'multiply', 'divide'],
        interactions_max_arithmetic_pairs=50,
        
        interactions_use_polynomial=True,
        interactions_polynomial_degree=2,
        interactions_polynomial_interaction_only=False,
        interactions_polynomial_max_features=5,  # Limit to prevent explosion
        
        interactions_use_ratios=True,
        interactions_ratios_include_proportions=True,
        interactions_ratios_include_log=False,
        interactions_max_ratio_pairs=30,
        
        interactions_use_products=False,  # 3-way products (optional)
        
        interactions_use_categorical_numeric=True,
        interactions_cat_num_strategy='both',  # group_stats + deviation
        interactions_max_cat_num_pairs=20,
        
        interactions_use_binned=False,  # Binned interactions (optional)
        
        # Other settings
        explain_transformations=True,
        explain_auto_print=False,
        random_state=42,
    )
    
    fe_interactions = AutoFeatureEngineer(config=config_interactions)
    
    print("\n   Fitting with feature interactions enabled...")
    fe_interactions.fit(X_train, y_train)
    
    X_train_interactions = fe_interactions.transform(X_train)
    X_test_interactions = fe_interactions.transform(X_test)
    
    print(f"\n   Features after preprocessing + interactions: {X_train_interactions.shape[1]}")
    print(f"   New features created: {X_train_interactions.shape[1] - X_train_baseline.shape[1]}")
    
    # Train models with interactions
    lr_interactions = LogisticRegression(max_iter=1000, random_state=42)
    lr_interactions.fit(X_train_interactions, y_train)
    
    rf_interactions = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_interactions.fit(X_train_interactions, y_train)
    
    # Evaluate
    lr_acc_interactions = accuracy_score(y_test, lr_interactions.predict(X_test_interactions))
    lr_auc_interactions = roc_auc_score(y_test, lr_interactions.predict_proba(X_test_interactions)[:, 1])
    
    rf_acc_interactions = accuracy_score(y_test, rf_interactions.predict(X_test_interactions))
    rf_auc_interactions = roc_auc_score(y_test, rf_interactions.predict_proba(X_test_interactions)[:, 1])
    
    print(f"\n   Logistic Regression - Accuracy: {lr_acc_interactions:.4f}, AUC: {lr_auc_interactions:.4f}")
    print(f"   Random Forest       - Accuracy: {rf_acc_interactions:.4f}, AUC: {rf_auc_interactions:.4f}")
    
    # ========== Comparison ==========
    print("\n" + "=" * 80)
    print("4. PERFORMANCE COMPARISON")
    print("=" * 80)
    
    print("\n   Logistic Regression (Linear Model):")
    print(f"      Baseline:     Accuracy={lr_acc_baseline:.4f}, AUC={lr_auc_baseline:.4f}")
    print(f"      Interactions: Accuracy={lr_acc_interactions:.4f}, AUC={lr_auc_interactions:.4f}")
    print(f"      Improvement:  Accuracy={lr_acc_interactions - lr_acc_baseline:+.4f}, AUC={lr_auc_interactions - lr_auc_baseline:+.4f}")
    
    print("\n   Random Forest (Non-linear Model):")
    print(f"      Baseline:     Accuracy={rf_acc_baseline:.4f}, AUC={rf_auc_baseline:.4f}")
    print(f"      Interactions: Accuracy={rf_acc_interactions:.4f}, AUC={rf_auc_interactions:.4f}")
    print(f"      Improvement:  Accuracy={rf_acc_interactions - rf_acc_baseline:+.4f}, AUC={rf_auc_interactions - rf_auc_baseline:+.4f}")
    
    # ========== Sample Feature Names ==========
    print("\n" + "=" * 80)
    print("5. SAMPLE GENERATED INTERACTION FEATURES")
    print("=" * 80)
    
    feature_names = X_train_interactions.columns.tolist()
    
    # Find interaction features
    arithmetic_features = [f for f in feature_names if any(op in f for op in ['_add_', '_sub_', '_mul_', '_div_'])]
    ratio_features = [f for f in feature_names if '_ratio_' in f or '_prop_' in f]
    poly_features = [f for f in feature_names if any(x in f for x in ['_^2', '_^3', 'poly_'])]
    cat_num_features = [f for f in feature_names if any(x in f for x in ['_mean', '_std', '_deviation', '_diff'])]
    
    print(f"\n   Total features: {len(feature_names)}")
    print(f"\n   Arithmetic interactions: {len(arithmetic_features)}")
    if arithmetic_features:
        print(f"      Examples: {arithmetic_features[:5]}")
    
    print(f"\n   Ratio/Proportion features: {len(ratio_features)}")
    if ratio_features:
        print(f"      Examples: {ratio_features[:5]}")
    
    print(f"\n   Polynomial features: {len(poly_features)}")
    if poly_features:
        print(f"      Examples: {poly_features[:5]}")
    
    print(f"\n   Categorical×Numeric interactions: {len(cat_num_features)}")
    if cat_num_features:
        print(f"      Examples: {cat_num_features[:5]}")
    
    # ========== Summary ==========
    print("\n" + "=" * 80)
    print("6. KEY TAKEAWAYS")
    print("=" * 80)
    
    print("\n   * Feature interactions capture non-linear relationships")
    print("   * Especially beneficial for LINEAR models (LR, SVM, etc.)")
    print("   * Tree-based models can learn interactions, but explicit features may help")
    print("   * Domain knowledge can guide specific interactions (e.g., debt/income ratio)")
    print("   * Use with caution: can create many features (consider dimensionality reduction)")
    
    print("\n   Interaction types implemented:")
    print("      - Arithmetic: A+B, A-B, A*B, A/B")
    print("      - Polynomial: x^2, x^3, x1*x2")
    print("      - Ratios: A/B, A/(A+B)")
    print("      - Products: A*B*C (multi-way)")
    print("      - Categorical*Numeric: group stats, deviations")
    print("      - Binned: bin then interact")
    
    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)


def demo_domain_specific_interactions():
    """Demonstrate domain-specific interaction use cases."""
    print("\n" + "=" * 80)
    print("7. DOMAIN-SPECIFIC INTERACTION EXAMPLES")
    print("=" * 80)
    
    # E-commerce example
    print("\n   E-commerce Example:")
    print("      price * quantity -> revenue")
    print("      clicks / impressions -> click_through_rate")
    print("      conversions / clicks -> conversion_rate")
    print("      cart_value / avg_order_value -> cart_size_ratio")
    
    # Healthcare example
    print("\n   Healthcare Example:")
    print("      weight / height^2 -> BMI")
    print("      systolic - diastolic -> pulse_pressure")
    print("      age * risk_score -> age_adjusted_risk")
    print("      medication_count * comorbidity_count -> complexity_score")

    # Finance example
    print("\n   Finance Example:")
    print("      debt / income -> debt_to_income_ratio")
    print("      savings / income -> savings_rate")
    print("      credit_score * income -> creditworthiness")
    print("      assets - liabilities -> net_worth")
    
    print("\n   These can be configured using:")
    print("      - interactions_specific_pairs: List[Tuple[str, str]]")
    print("      - interactions_domain_formulas: Dict[str, str]")


if __name__ == "__main__":
    demo_all_interactions()
    demo_domain_specific_interactions()
    
    print("\n" + "=" * 80)
    print("For more examples, see: https://github.com/yourusername/featurecraft")
    print("=" * 80)

