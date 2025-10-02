
"""
@01_quickstart.py

Simple demonstration of FeatureCraft library.
Shows how to use automated feature engineering on sample datasets.
"""

import warnings
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.datasets import load_iris, load_wine

warnings.filterwarnings('ignore')


def main():
    """Simple demonstration of FeatureCraft library."""
    print("=" * 50)
    print("FeatureCraft Library - Quick Start Demo")
    print("=" * 50)

    # Load sample datasets
    print("\n1. Loading sample datasets...")

    # Load Iris dataset (classification)
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_target = pd.Series(iris.target, name='species')

    # Load Wine dataset (classification)
    wine = load_wine()
    wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
    wine_target = pd.Series(wine.target, name='wine_type')

    print(f"   Iris dataset: {iris_df.shape[0]} samples, {iris_df.shape[1]} features")
    print(f"   Wine dataset: {wine_df.shape[0]} samples, {wine_df.shape[1]} features")

    # Initialize FeatureCraft
    print("\n2. Initializing FeatureCraft...")
    try:
        from featurecraft.pipeline import AutoFeatureEngineer
        fe = AutoFeatureEngineer()
        print("   [OK] FeatureCraft initialized successfully")
    except ImportError as e:
        print(f"   [ERROR] Could not import FeatureCraft: {e}")
        print("   Please install with: pip install featurecraft")
        return

    # Process Iris dataset
    print("\n3. Processing Iris dataset...")
    iris_X_train, iris_X_test, iris_y_train, iris_y_test = train_test_split(
        iris_df, iris_target, test_size=0.2, random_state=42, stratify=iris_target
    )

    # Fit and transform
    fe.fit(iris_X_train, iris_y_train)
    iris_X_train_fe = fe.transform(iris_X_train)
    iris_X_test_fe = fe.transform(iris_X_test)

    print(f"   Original features: {iris_X_train.shape[1]}")
    print(f"   Engineered features: {iris_X_train_fe.shape[1]}")

    # Train and evaluate
    clf = RandomForestClassifier(random_state=42, n_estimators=10)
    clf.fit(iris_X_train_fe, iris_y_train)
    iris_pred = clf.predict(iris_X_test_fe)
    iris_acc = accuracy_score(iris_y_test, iris_pred)

    print(f"   Test accuracy: {iris_acc:.3f}")

    # Process Wine dataset
    print("\n4. Processing Wine dataset...")
    wine_X_train, wine_X_test, wine_y_train, wine_y_test = train_test_split(
        wine_df, wine_target, test_size=0.2, random_state=42, stratify=wine_target
    )

    # Fit and transform
    fe.fit(wine_X_train, wine_y_train)
    wine_X_train_fe = fe.transform(wine_X_train)
    wine_X_test_fe = fe.transform(wine_X_test)

    print(f"   Original features: {wine_X_train.shape[1]}")
    print(f"   Engineered features: {wine_X_train_fe.shape[1]}")
                
                # Train and evaluate
    clf = RandomForestClassifier(random_state=42, n_estimators=10)
    clf.fit(wine_X_train_fe, wine_y_train)
    wine_pred = clf.predict(wine_X_test_fe)
    wine_acc = accuracy_score(wine_y_test, wine_pred)

    print(f"   Test accuracy: {wine_acc:.3f}")

    print("\n" + "=" * 50)
    print("FeatureCraft Demo Complete!")
    print("=" * 50)
    print("\nKey takeaways:")
    print("- FeatureCraft automatically creates new features")
    print("- Engineered features can improve model performance")
    print("- Works with different types of datasets")
    print("\nFor more advanced usage, see the documentation.")


if __name__ == "__main__":
    main()