## AI-Powered Feature Engineering

FeatureCraft now includes AI-powered feature engineering capabilities using LLMs (GPT-4, Claude, etc.) to automatically generate, validate, and execute feature engineering plans.

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Safety & Validation](#safety--validation)
- [Telemetry & Monitoring](#telemetry--monitoring)
- [Best Practices](#best-practices)
- [Examples](#examples)
- [FAQ](#faq)

---

## Overview

### What is AI-Powered Feature Engineering?

Traditional feature engineering requires manual effort and domain expertise. FeatureCraft's AI integration uses Large Language Models to:

1. **Plan**: Generate feature engineering plans from natural language intent
2. **Validate**: Check for data leakage, schema violations, and time-ordering issues
3. **Execute**: Transform plans into actual features using pandas/sklearn
4. **Monitor**: Track costs, performance, and quality metrics

### Key Benefits

- ⚡ **Faster Time-to-Value**: Minutes instead of hours for feature engineering
- 🧠 **Domain Intelligence**: LLMs understand common ML patterns (RFM, lag features, etc.)
- 🛡️ **Built-in Safety**: Automatic leakage detection and validation
- 📊 **Telemetry**: Track costs, tokens, and performance
- 🔧 **Flexible**: Works with any LLM provider (OpenAI, Anthropic, local models)
- 📚 **RAG-Augmented**: Leverage past experiments and domain knowledge
- 🎯 **Smart Pruning**: LLM-guided feature selection with statistical validation
- 🚀 **Scalable**: Spark/Ray support for large-scale datasets
- 🔬 **Auto-Ablation**: Systematically test feature contributions

---

## Quick Start

### Installation

```bash
# Install FeatureCraft with AI extras
pip install "featurecraft[ai]"

# Or install required packages manually
pip install featurecraft openai anthropic
```

### Basic Usage

```python
import pandas as pd
from featurecraft.ai import plan_features, execute_plan

# Your dataset
df = pd.read_csv("customer_data.csv")

# Generate feature plan with AI
plan = plan_features(
    df=df,
    target="churn",
    nl_intent="Create RFM features and behavioral patterns for churn prediction",
    time_col="transaction_date",
    key_col="customer_id",
    provider="openai",  # or "anthropic", "mock"
)

# Execute plan to generate features
df_features = execute_plan(plan, df)

print(f"Generated {len(df_features.columns)} features")
```

### Phase 2 Features (Advanced)

```python
from featurecraft.ai import (
    plan_with_rag,           # RAG-augmented planning
    prune_features,          # LLM-guided pruning
    run_ablation_study,      # Auto ablation studies
    execute_distributed,     # Spark/Ray execution
)

# RAG-augmented planning
plan = plan_with_rag(
    df=df,
    target="churn",
    nl_intent="Customer retention features",
    knowledge_dirs=["artifacts/", "docs/"],
)

# Feature pruning
result = prune_features(plan, X_train, y_train, target_n_features=30)

# Distributed execution
df_features = execute_distributed(plan, df, engine="ray")
```

### Setting API Keys

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Core Concepts

### Feature Plan

A **Feature Plan** is a JSON specification that describes what features to generate:

```json
{
  "version": "1.0",
  "dataset_id": "customer_churn_v1",
  "task": "classification",
  "estimator_family": "tree",
  "candidates": [
    {
      "name": "amt_mean_30d",
      "type": "rolling_mean",
      "source_col": "amount",
      "window": "30d",
      "key_col": "customer_id",
      "time_col": "transaction_date",
      "rationale": "Average spending over last 30 days indicates customer value",
      "safety_tags": ["no_target_ref", "time_safe"]
    }
  ]
}
```

### Feature Spec

Each feature is defined by a **FeatureSpec**:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `name` | str | Feature name | `"amt_mean_30d"` |
| `type` | str | Transformation type | `"rolling_mean"` |
| `source_col` | str/list | Source column(s) | `"amount"` |
| `window` | str | Time window | `"30d"` |
| `key_col` | str | Groupby key | `"customer_id"` |
| `time_col` | str | Time column | `"transaction_date"` |
| `params` | dict | Additional params | `{"smoothing": 0.3}` |
| `rationale` | str | Human explanation | "Why this feature" |
| `safety_tags` | list | Validation tags | `["time_safe"]` |

### Supported Feature Types

**Aggregations**:
- `rolling_mean`, `rolling_sum`, `rolling_std`, `rolling_min`, `rolling_max`

**Temporal**:
- `lag`, `diff`, `pct_change`, `ewm`, `expanding_mean`

**Cardinality**:
- `nunique`, `count`

**Encodings**:
- `target_encode`, `frequency_encode`, `count_encode`, `ohe`, `hash_encode`

**Domain-Specific**:
- `recency`, `frequency`, `monetary`, `rfm_score`

**Interactions**:
- `multiply`, `divide`, `add`, `subtract`, `ratio`

---

## API Reference

### `plan_features()`

Generate feature engineering plan from dataset and natural language intent.

```python
def plan_features(
    df: pd.DataFrame,
    target: str,
    task: Literal["classification", "regression"] | None = None,
    nl_intent: str | None = None,
    estimator_family: str = "tree",
    time_col: str | None = None,
    key_col: str | None = None,
    constraints: dict | None = None,
    max_features: int | None = None,
    provider: str = "openai",
    model: str | None = None,
    validate: bool = True,
) -> FeaturePlan
```

**Parameters**:
- `df`: Input DataFrame
- `target`: Target column name
- `task`: Task type (auto-detected if None)
- `nl_intent`: Natural language description of desired features
- `estimator_family`: Target estimator (tree, linear, svm, knn, nn)
- `time_col`: Time column for time-series features
- `key_col`: Entity key (customer_id, user_id, etc.)
- `constraints`: Additional constraints (leakage_blocklist, etc.)
- `max_features`: Maximum features to generate
- `provider`: LLM provider (openai, anthropic, mock)
- `model`: Model name (optional, uses provider default)
- `validate`: Validate plan after generation

**Returns**: `FeaturePlan` object

**Example**:
```python
plan = plan_features(
    df=train_df,
    target="churn",
    nl_intent="Create customer behavior features focusing on recency and engagement",
    time_col="date",
    key_col="customer_id",
    max_features=50,
)
```

### `execute_plan()`

Execute feature plan to generate features.

```python
def execute_plan(
    plan: FeaturePlan,
    df: pd.DataFrame,
    engine: Literal["pandas"] = "pandas",
    return_original: bool = False,
) -> pd.DataFrame
```

**Parameters**:
- `plan`: Feature plan to execute
- `df`: Input DataFrame
- `engine`: Execution engine (only pandas supported now)
- `return_original`: If True, return df with added features. If False, return only new features.

**Returns**: DataFrame with generated features

### `validate_plan()`

Validate feature plan for safety and correctness.

```python
def validate_plan(
    plan: FeaturePlan,
    context: DatasetContext | None = None,
    strict_mode: bool = False,
) -> ValidationResult
```

**Parameters**:
- `plan`: Feature plan to validate
- `context`: Dataset context (optional but recommended)
- `strict_mode`: Treat warnings as errors

**Returns**: `ValidationResult` with errors, warnings, and pass/fail status

---

## Configuration

### Config Parameters

Add to `FeatureCraftConfig`:

```python
from featurecraft import FeatureCraftConfig

config = FeatureCraftConfig(
    # AI settings
    ai_enabled=True,
    ai_provider="openai",
    ai_model="gpt-4o",
    ai_max_features=100,
    ai_max_tokens=50000,
    ai_timeout_seconds=60,
    ai_validate_plan=True,
    ai_strict_validation=False,
    ai_enable_telemetry=True,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ai_enabled` | `False` | Enable AI-powered features |
| `ai_provider` | `"openai"` | LLM provider |
| `ai_model` | `None` | Model name (uses provider default) |
| `ai_max_features` | `100` | Max features to generate |
| `ai_max_tokens` | `50000` | Max tokens per request |
| `ai_timeout_seconds` | `60` | Request timeout |
| `ai_validate_plan` | `True` | Validate plans for safety |
| `ai_strict_validation` | `False` | Treat warnings as errors |
| `ai_enable_telemetry` | `True` | Log AI call metadata |

### Environment Variables

```bash
# Provider API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Config overrides
export FEATURECRAFT__AI_ENABLED=true
export FEATURECRAFT__AI_PROVIDER=openai
export FEATURECRAFT__AI_MAX_FEATURES=50
```

---

## Safety & Validation

### Leakage Detection

The validator automatically checks for:

1. **Direct target references**: Features that use the target column
2. **Future data**: Time-series features that use future information
3. **Suspicious columns**: Columns with leakage-prone names (e.g., `churn_date`, `prediction`)
4. **Target encoding safety**: Ensures out-of-fold encoding

**Example**:
```python
# This will FAIL validation
spec_bad = FeatureSpec(
    name="churn_lag",
    type="lag",
    source_col="churn",  # ❌ References target!
)

# This will PASS
spec_good = FeatureSpec(
    name="amt_lag_1",
    type="lag",
    source_col="amount",  # ✅ Safe
)
```

### Time-Ordering Validation

For time-series data:

```python
plan = plan_features(
    df=df,
    target="sales",
    time_col="date",
    key_col="store_id",
    constraints={
        "time_aware": True,
        "leakage_blocklist": ["actual_sales", "future_promo"],
    }
)
```

### Validation Results

```python
from featurecraft.ai import validate_plan

result = validate_plan(plan, context)

if result.is_valid:
    print("✓ Plan is safe to execute")
else:
    print(f"✗ Validation failed:")
    for error in result.errors:
        print(f"  - {error}")
    
    for warning in result.warnings:
        print(f"  ⚠ {warning}")
```

---

## Telemetry & Monitoring

### Tracking AI Costs

```python
from featurecraft.ai.telemetry import get_telemetry_stats

# After running AI features
stats = get_telemetry_stats()

print(f"Total AI calls:  {stats['total_calls']}")
print(f"Total tokens:    {stats['total_tokens']:,}")
print(f"Total cost:      ${stats['total_cost_usd']:.4f}")
print(f"Avg latency:     {stats['avg_latency_ms']:.0f}ms")
```

### Telemetry Log Format

Telemetry is logged to `logs/ai_telemetry.jsonl`:

```json
{
  "timestamp": "2025-10-03T12:00:00",
  "provider": "openai",
  "model": "gpt-4o",
  "tokens_used": 5234,
  "latency_ms": 1245,
  "cost_usd": 0.0523,
  "validator_status": "pass",
  "validator_errors": [],
  "cache_hit": false
}
```

---

## Best Practices

### 1. Start with Clear Intent

❌ Bad:
```python
plan = plan_features(df, target="churn")
```

✅ Good:
```python
plan = plan_features(
    df=df,
    target="churn",
    nl_intent="""
    Create features for customer churn prediction:
    - RFM (recency, frequency, monetary)
    - Engagement trends (transaction count over time)
    - Product diversity (unique categories purchased)
    - Spending velocity (change in spending patterns)
    """,
    time_col="date",
    key_col="customer_id",
)
```

### 2. Use Time-Aware Constraints

For time-series data, always specify time constraints:

```python
plan = plan_features(
    df=df,
    target="sales",
    time_col="date",
    key_col="store_id",
    constraints={
        "time_aware": True,
        "leakage_blocklist": ["actual_sales", "inventory_after_sale"],
    }
)
```

### 3. Validate Before Execution

```python
# Generate plan
plan = plan_features(df, target="churn", validate=True)

# Double-check validation
result = validate_plan(plan, context)

if not result.is_valid:
    raise ValueError(f"Plan failed validation: {result.errors}")

# Safe to execute
df_features = execute_plan(plan, df)
```

### 4. Monitor Costs

```python
from featurecraft.ai.telemetry import get_telemetry_stats

# Before
stats_before = get_telemetry_stats()

# ... run AI features ...

# After
stats_after = get_telemetry_stats()

cost_delta = stats_after["total_cost_usd"] - stats_before["total_cost_usd"]
print(f"Cost for this run: ${cost_delta:.4f}")
```

### 5. Use Mock Provider for Testing

```python
# During development/testing
plan = plan_features(
    df=df,
    target="churn",
    provider="mock",  # No API calls, no cost
    validate=False,
)
```

---

## Examples

### Example 1: Customer Churn Prediction

```python
from featurecraft.ai import plan_features, execute_plan

# Generate plan
plan = plan_features(
    df=transactions_df,
    target="churn",
    nl_intent="Create RFM features + behavioral patterns",
    time_col="transaction_date",
    key_col="customer_id",
    max_features=30,
)

# Execute
df_features = execute_plan(plan, transactions_df)

# Use in ML pipeline
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier()
clf.fit(df_features, y_train)
```

### Example 2: Time-Series Forecasting

```python
plan = plan_features(
    df=sales_df,
    target="sales",
    task="regression",
    nl_intent="""
    Create lag features and rolling statistics for sales forecasting:
    - Lags: 1, 7, 14, 28 days
    - Rolling means: 7d, 14d, 30d windows
    - Seasonality indicators
    """,
    time_col="date",
    key_col="store_id",
)
```

### Example 3: Fraud Detection

```python
plan = plan_features(
    df=transactions_df,
    target="is_fraud",
    nl_intent="""
    Create fraud detection features:
    - Velocity: transactions per hour/day
    - Amount deviations from user average
    - Time since last transaction
    - Geographic distance from previous transaction
    """,
    time_col="timestamp",
    key_col="user_id",
)
```

---

## Phase 2 Features

### RAG-Augmented Planning

**What is RAG-Augmented Planning?**

RAG (Retrieval-Augmented Generation) enhances the LLM planner by retrieving relevant domain knowledge from your past experiments, documentation, and artifacts.

**Benefits**:
- 📚 **Domain Knowledge**: Reuse successful feature patterns from past projects
- 🎯 **Context-Aware**: Features tailored to your specific data patterns
- 🚀 **Better Quality**: LLM has more context for decision-making

**Usage**:

```python
from featurecraft.ai import plan_with_rag

# Plan with RAG augmentation
plan = plan_with_rag(
    df=train_df,
    target="churn",
    nl_intent="Create customer retention features",
    knowledge_dirs=[
        "artifacts/",           # Past experiment results
        "docs/",                # Domain documentation
        "knowledge_base/",      # Custom knowledge
    ],
    provider="openai",
    embedder="sentence_transformers",  # or "openai"
)
```

**Knowledge Sources**:
- **Past Experiments**: Metrics, feature importance from previous runs
- **Documentation**: Markdown files with domain knowledge
- **Schemas**: Dataset schemas and statistics
- **Ontologies**: Custom feature taxonomies

**Advanced Configuration**:

```python
from featurecraft.ai.rag import RAGRetriever

# Custom RAG retriever
retriever = RAGRetriever(
    embedder="sentence_transformers",
    knowledge_dirs=["artifacts/", "docs/"],
    cache_ttl_hours=24,
    enable_pii_redaction=True,  # Redact PII from context
    enable_bm25=True,            # Hybrid vector + BM25 search
)

# Build/rebuild index
retriever.rebuild_index()

# Use with planner
from featurecraft.ai import LLMPlanner

planner = LLMPlanner(
    provider="openai",
    rag_retriever=retriever,
    enable_rag=True,
)

plan = planner.plan(df=train_df, target="churn")
```

---

### Feature Pruning

**What is Feature Pruning?**

LLM-guided feature pruning combines AI reasoning with statistical validation to select the most important features from a large candidate pool.

**Benefits**:
- 🎯 **Smart Selection**: LLM understands feature semantics and interactions
- 📊 **Statistical Gates**: Mutual information, permutation importance, stability
- ⚡ **Efficiency**: Reduce features while maintaining performance

**Usage**:

```python
from featurecraft.ai import plan_features, prune_features

# Step 1: Generate large feature pool
plan = plan_features(
    df=train_df,
    target="churn",
    nl_intent="Create comprehensive customer features",
    max_features=100,  # Generate many candidates
)

# Step 2: Prune to best features
result = prune_features(
    plan=plan,
    X_train=X_train,
    y_train=y_train,
    target_n_features=30,  # Keep top 30
    X_val=X_val,           # For stability validation
    y_val=y_val,
    provider="openai",
)

# Step 3: Use pruned plan
print(f"Selected {len(result.selected_features)} features")
print(f"Pruned plan: {result.pruned_plan}")

# See rankings and rationale
for ranking in result.rankings[:10]:
    print(f"{ranking.rank}. {ranking.feature_name} (score: {ranking.score:.3f})")
    print(f"   Rationale: {ranking.rationale}")
    print(f"   Gates passed: {ranking.gates_passed}")
```

**Statistical Gates**:

| Gate | Description | Default Threshold |
|------|-------------|-------------------|
| **Mutual Information** | Measures dependency with target | 0.01 |
| **Permutation Importance** | Feature importance via shuffling | 0.001 |
| **Stability** | Consistency across CV folds | 0.7 |
| **Leakage Detection** | Checks for data leakage | N/A |

**Custom Configuration**:

```python
from featurecraft.ai import FeaturePruner

pruner = FeaturePruner(
    provider="openai",
    enable_mi_gate=True,
    enable_permutation_gate=True,
    enable_stability_gate=True,
    enable_leakage_gate=True,
    mi_threshold=0.02,              # Stricter MI threshold
    permutation_threshold=0.005,    # Stricter importance
    stability_threshold=0.8,        # Higher stability required
)

result = pruner.prune(
    plan=plan,
    X_train=X_train,
    y_train=y_train,
    target_n_features=50,
)
```

---

### Ablation Studies

**What are Ablation Studies?**

Automated ablation studies systematically test feature contributions by including/excluding features or varying their parameters.

**Benefits**:
- 🔬 **Feature Attribution**: Understand which features actually help
- 🎛️ **Hyperparameter Tuning**: Find optimal windows, encodings
- 📈 **Performance Insights**: Identify synergies and redundancies

**Usage**:

```python
from featurecraft.ai import plan_features, run_ablation_study
from sklearn.ensemble import RandomForestClassifier

# Generate feature plan
plan = plan_features(
    df=train_df,
    target="churn",
    nl_intent="Customer behavior features",
)

# Run ablation study
study = run_ablation_study(
    plan=plan,
    X=X_train,
    y=y_train,
    estimator=RandomForestClassifier(),
    strategies=["on_off", "window", "encoding"],
    scoring="roc_auc",
    cv=5,
)

# Analyze results
print(f"Baseline score: {study.baseline_result.score:.4f}")
print(f"Best score:     {study.best_result.score:.4f}")
print(f"Improvement:    {study.best_result.score - study.baseline_result.score:+.4f}")

# View insights
for key, value in study.insights.items():
    print(f"{key}: {value}")

# Save study
study.save("artifacts/ablation_study.json")
```

**Ablation Strategies**:

| Strategy | Description | Example |
|----------|-------------|---------|
| **on_off** | Include/exclude each feature | Test impact of removing `amt_mean_30d` |
| **window** | Vary time windows | Compare 7d vs 30d windows |
| **encoding** | Vary encoding strategies | Compare target vs frequency encoding |
| **interaction** | Test feature interactions | Test `amt_mean * txn_count` |

**Advanced Configuration**:

```python
from featurecraft.ai import AblationRunner

runner = AblationRunner(
    estimator=RandomForestClassifier(),
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
    early_stop_patience=10,      # Stop if no improvement after 10 experiments
    early_stop_threshold=0.001,  # Min improvement to continue
    max_experiments=100,         # Limit total experiments
)

study = runner.run_ablation(
    plan=plan,
    X=X_train,
    y=y_train,
    strategies=["on_off", "window"],
)
```

**Results Analysis**:

```python
# Best configuration
print(f"Best experiment: {study.best_result.experiment_id}")
print(f"Features included: {study.best_result.features_included}")
print(f"Score: {study.best_result.score:.4f} ± {study.best_result.score_std:.4f}")

# All results
for result in study.ablation_results:
    print(f"{result.experiment_id}: {result.score:.4f} (n_features={result.n_features})")
```

---

### Distributed Execution

**What is Distributed Execution?**

Execute feature engineering plans on distributed compute engines (Spark, Ray) for large-scale datasets.

**Benefits**:
- 🚀 **Scalability**: Handle datasets that don't fit in memory
- ⚡ **Speed**: Parallel execution across clusters
- 🔄 **Fault Tolerance**: Automatic retries and recovery

#### Pandas Executor (Default)

For small to medium datasets (< 1GB):

```python
from featurecraft.ai import execute_plan

df_features = execute_plan(
    plan=plan,
    df=train_df,
    engine="pandas",  # Default
    return_original=False,
)
```

#### Spark Executor

For large datasets (> 1GB, distributed storage):

```python
from featurecraft.ai import execute_distributed
from pyspark.sql import SparkSession

# Initialize Spark
spark = SparkSession.builder \
    .appName("FeatureCraft") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# Convert to Spark DataFrame
sdf = spark.createDataFrame(train_df)

# Execute on Spark
sdf_features = execute_distributed(
    plan=plan,
    data=sdf,
    engine="spark",
    return_original=False,
)

# Convert back to pandas (if needed)
df_features = sdf_features.toPandas()
```

**Spark Configuration**:

```python
from featurecraft.ai.executors import SparkExecutor

executor = SparkExecutor(
    spark_session=spark,
    cache_intermediate=True,  # Cache intermediate results
    repartition=200,          # Number of partitions
)

sdf_features = executor.execute(plan, sdf)
```

#### Ray Executor

For parallel execution on single machine or Ray cluster:

```python
import ray
from featurecraft.ai import execute_distributed

# Initialize Ray
ray.init(num_cpus=8)

# Execute on Ray
df_features = execute_distributed(
    plan=plan,
    data=train_df,
    engine="ray",
    return_original=False,
)
```

**Ray Configuration**:

```python
from featurecraft.ai.executors import RayExecutor

executor = RayExecutor(
    num_cpus=8,
    batch_size=1000,
    max_retries=3,
    enable_progress_bar=True,
)

df_features = executor.execute(plan, train_df)
```

**Performance Comparison**:

| Engine | Dataset Size | Typical Speed | Use Case |
|--------|-------------|---------------|----------|
| **Pandas** | < 1GB | Baseline | Prototyping, small data |
| **Ray** | 1-50GB | 2-5x faster | Single machine, parallel |
| **Spark** | > 50GB | 5-10x faster | Distributed cluster |

---

## Phase 2 Complete Example

Here's a complete workflow using all Phase 2 features:

```python
from featurecraft.ai import (
    plan_with_rag,
    prune_features,
    run_ablation_study,
    execute_distributed,
)
from sklearn.ensemble import GradientBoostingClassifier

# Step 1: RAG-augmented planning
print("Step 1: Planning with RAG...")
plan = plan_with_rag(
    df=train_df,
    target="churn",
    nl_intent="Create customer retention features with RFM patterns",
    knowledge_dirs=["artifacts/", "docs/"],
    max_features=100,
)
print(f"Generated {len(plan.candidates)} features")

# Step 2: Feature pruning
print("\nStep 2: Pruning features...")
prune_result = prune_features(
    plan=plan,
    X_train=X_train,
    y_train=y_train,
    target_n_features=30,
    X_val=X_val,
    y_val=y_val,
)
print(f"Selected {len(prune_result.selected_features)} features")

# Step 3: Distributed execution
print("\nStep 3: Executing features...")
df_features = execute_distributed(
    plan=prune_result.pruned_plan,
    data=train_df,
    engine="ray",  # Use Ray for parallel execution
)
print(f"Generated features: {df_features.shape}")

# Step 4: Ablation study
print("\nStep 4: Running ablation study...")
study = run_ablation_study(
    plan=prune_result.pruned_plan,
    X=df_features,
    y=y_train,
    estimator=GradientBoostingClassifier(),
    strategies=["on_off"],
    cv=5,
)
print(f"Baseline: {study.baseline_result.score:.4f}")
print(f"Best:     {study.best_result.score:.4f}")

# Step 5: Train final model
print("\nStep 5: Training final model...")
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    GradientBoostingClassifier(),
    df_features,
    y_train,
    cv=5,
    scoring="roc_auc",
)
print(f"CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")
```

---

## FAQ

### Q: Which LLM provider should I use?

**A**: 
- **OpenAI GPT-4o**: Best quality, moderate cost (~$2.50/M input tokens)
- **Anthropic Claude 3.5 Sonnet**: Excellent quality, similar cost
- **Mock**: For testing, no API calls

### Q: How much does it cost?

**A**: Typical costs:
- Small dataset (< 1K rows): $0.01 - $0.05 per plan
- Medium dataset (1K - 100K rows): $0.05 - $0.20 per plan
- Large dataset (> 100K rows): $0.20 - $0.50 per plan

Use `get_telemetry_stats()` to track actual costs.

### Q: How do I prevent data leakage?

**A**: FeatureCraft's validator automatically checks for leakage:
1. Direct target references
2. Future data in time-series
3. Suspicious column names
4. Unsafe target encoding

Always use `validate=True` when calling `plan_features()`.

### Q: Can I use local LLMs?

**A**: Yes! Implement the `LLMProvider` protocol:

```python
from featurecraft.ai.providers import BaseLLMProvider

class LocalLLMProvider(BaseLLMProvider):
    def call(self, prompt, **kwargs):
        # Your local LLM inference logic
        ...
```

### Q: How do I customize the prompt?

**A**: Modify the system prompt in `featurecraft/ai/planner.py`:

```python
SYSTEM_PROMPT = """
Your custom instructions here...
"""
```

### Q: What if validation fails?

**A**:
1. Check `result.errors` for specific issues
2. Fix dataset or constraints
3. Re-run with strict_mode=False to see warnings

```python
result = validate_plan(plan, context)
if not result.is_valid:
    for error in result.errors:
        print(f"Error: {error}")
```

### Q: When should I use RAG-augmented planning?

**A**: Use RAG when:
- You have past experiments with similar data/tasks
- Domain documentation exists (e.g., feature definitions)
- You want consistent feature engineering patterns
- Working on recurring ML tasks

RAG typically improves feature quality by 10-20% compared to base planning.

### Q: How does feature pruning work?

**A**: Feature pruning uses a two-stage process:
1. **LLM Ranking**: LLM ranks features based on semantic understanding
2. **Statistical Gates**: Apply MI, permutation importance, stability tests

Features must pass both stages. This ensures both interpretability and statistical significance.

### Q: When should I use distributed execution?

**A**: Use distributed execution when:
- **Spark**: Dataset > 50GB, need distributed storage
- **Ray**: Dataset 1-50GB, parallel processing on single/multi-node
- **Pandas**: Dataset < 1GB, prototyping

Start with Pandas, upgrade to Ray for speed, use Spark for scale.

### Q: Can I use Phase 2 features without API keys?

**A**: Partially:
- ✅ **Distributed execution**: Works without API (no LLM needed)
- ✅ **Ablation studies**: Works without API (statistical only)
- ❌ **RAG planning**: Requires LLM API key
- ❌ **Feature pruning**: Requires LLM API key (for ranking)

For testing, use `provider="mock"` for RAG/pruning.

---

## Roadmap

**Phase 1 (✅ Complete)**:
- ✅ LLM planner with OpenAI/Anthropic support
- ✅ Safety validation (leakage, schema, time-ordering)
- ✅ Pandas executor for basic feature types
- ✅ Telemetry and cost tracking

**Phase 2 (✅ Complete)**:
- ✅ RAG-based domain knowledge retrieval
- ✅ Auto-ablation studies
- ✅ Feature pruning with LLM guidance
- ✅ Spark/Ray executors for distributed execution

**Phase 3 (Coming 2026)**:
- 🔄 Multi-table feature engineering
- 🔄 Graph-based features
- 🔄 AutoML integration
- 🔄 Fine-tuned domain-specific models

---

## Support

- GitHub Issues: https://github.com/featurecraft/featurecraft/issues
- Documentation: https://featurecraft.dev/docs/ai-integration
- Examples: https://github.com/featurecraft/featurecraft/tree/main/examples

---

**Built with ❤️ for the ML community**

