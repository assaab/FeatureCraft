# FeatureCraft CLI Reference

Complete reference for FeatureCraft's command-line interface.

## Table of Contents

- [Installation & Setup](#installation--setup)
- [Global Options](#global-options)
- [Commands](#commands)
  - [analyze](#analyze)
  - [fit](#fit)
  - [fit-transform](#fit-transform)
  - [transform](#transform)
  - [print-config](#print-config)
  - [wizard](#wizard)
- [Configuration](#configuration)
- [Examples](#examples)

---

## Installation & Setup

```bash
# Install FeatureCraft
pip install featurecraft

# Verify installation
featurecraft --version

# Get help
featurecraft --help
```

---

## Global Options

These options work with all commands:

```bash
featurecraft COMMAND [OPTIONS]

Global Options:
  --version              Show version and exit
  --help                 Show help message
  --config PATH          Path to config file (YAML/JSON/TOML)
  --set KEY=VALUE        Override config parameter (can be used multiple times)
  --random-state INT     Set random seed
  --verbose, -v          Increase verbosity (use -vv for debug)
  --quiet, -q            Suppress output
```

---

## Commands

### `analyze`

Analyze dataset and generate insights report.

#### Usage

```bash
featurecraft analyze --input DATA.csv --target TARGET [OPTIONS]
```

#### Required Arguments

- `--input PATH`, `-i PATH`: Input CSV/parquet file
- `--target NAME`, `-t NAME`: Target column name

#### Optional Arguments

```bash
--out DIR, -o DIR                Output directory (default: artifacts)
--reference PATH                 Reference dataset for drift detection
--open                          Open HTML report in browser
--format {html,json}            Report format (default: html)
```

#### Configuration Overrides

```bash
--set enable_drift_detection=true
--set drift_psi_threshold=0.25
--set verbosity=2
```

#### Examples

```bash
# Basic analysis
featurecraft analyze --input data.csv --target target --out artifacts/

# With drift detection
featurecraft analyze \
  --input production.csv \
  --target target \
  --reference training.csv \
  --set enable_drift_detection=true \
  --open

# With custom config
featurecraft analyze \
  --input data.csv \
  --target target \
  --config my_config.yaml \
  --out results/
```

#### Output

- `{out}/report.html`: Interactive HTML report
- `{out}/insights.json`: Analysis results (if `--format json`)
- Console: Summary statistics

---

### `fit`

Fit feature engineering pipeline.

#### Usage

```bash
featurecraft fit --input TRAIN.csv --target TARGET [OPTIONS]
```

#### Required Arguments

- `--input PATH`, `-i PATH`: Training CSV/parquet file
- `--target NAME`, `-t NAME`: Target column name

#### Optional Arguments

```bash
--out DIR, -o DIR                    Output directory (default: artifacts)
--estimator-family {tree,linear,svm,knn,nn}
                                     Model family (default: tree)
--groups COLUMN                      Column for GroupKFold CV
--config PATH                        Config file path
```

#### Configuration Overrides

```bash
--set low_cardinality_max=15
--set use_smote=true
--set smote_threshold=0.10
--random-state 42
```

#### Examples

```bash
# Basic fit
featurecraft fit --input train.csv --target target --out pipeline/

# For linear models with SMOTE
featurecraft fit \
  --input train.csv \
  --target target \
  --estimator-family linear \
  --set use_smote=true \
  --set smote_threshold=0.10 \
  --out pipeline/

# With custom config
featurecraft fit \
  --input train.csv \
  --target target \
  --config prod_config.yaml \
  --out pipeline/
```

#### Output

- `{out}/pipeline.joblib`: Fitted sklearn Pipeline
- `{out}/metadata.json`: Configuration and summary
- `{out}/feature_names.txt`: Output feature names

---

### `fit-transform`

Fit pipeline and transform training data.

#### Usage

```bash
featurecraft fit-transform --input TRAIN.csv --target TARGET [OPTIONS]
```

#### Required Arguments

Same as `fit` command.

#### Optional Arguments

```bash
--out DIR, -o DIR                    Output directory (default: artifacts)
--estimator-family {tree,linear,svm,knn,nn}
                                     Model family (default: tree)
--output-data PATH                   Save transformed data to this path
--groups COLUMN                      Column for GroupKFold CV
--config PATH                        Config file path
```

#### Examples

```bash
# Fit and save transformed data
featurecraft fit-transform \
  --input train.csv \
  --target target \
  --output-data train_transformed.csv \
  --out pipeline/

# For tree models with custom config
featurecraft fit-transform \
  --input train.csv \
  --target target \
  --estimator-family tree \
  --config tree_config.yaml \
  --output-data train_transformed.csv \
  --out pipeline/
```

#### Output

Same as `fit`, plus:
- `{output-data}`: Transformed training data (if specified)

---

### `transform`

Transform data using fitted pipeline.

#### Usage

```bash
featurecraft transform --input TEST.csv --pipeline PIPELINE.joblib [OPTIONS]
```

#### Required Arguments

- `--input PATH`, `-i PATH`: Input CSV/parquet file
- `--pipeline PATH`, `-p PATH`: Path to fitted pipeline.joblib

#### Optional Arguments

```bash
--output PATH, -o PATH              Output CSV/parquet file
--format {csv,parquet}              Output format (default: csv)
```

#### Examples

```bash
# Transform test data
featurecraft transform \
  --input test.csv \
  --pipeline artifacts/pipeline.joblib \
  --output test_transformed.csv

# Transform with parquet output
featurecraft transform \
  --input test.parquet \
  --pipeline pipeline.joblib \
  --output test_transformed.parquet \
  --format parquet
```

#### Output

- `{output}`: Transformed data
- Console: Transformation summary

---

### `print-config`

Print effective configuration.

#### Usage

```bash
featurecraft print-config [OPTIONS]
```

#### Optional Arguments

```bash
--config PATH                        Load config from file
--format {yaml,json,toml}            Output format (default: yaml)
--schema                             Print JSON schema instead
--set KEY=VALUE                      Apply overrides before printing
```

#### Examples

```bash
# Print default config
featurecraft print-config

# Print with custom config and overrides
featurecraft print-config \
  --config prod_config.yaml \
  --set use_smote=true \
  --format yaml

# Export JSON schema
featurecraft print-config --schema > config_schema.json

# Save merged config
featurecraft print-config \
  --config base.yaml \
  --set random_state=42 \
  --format yaml > final_config.yaml
```

#### Output

Console: Configuration in requested format

---

### `wizard`

Interactive configuration wizard.

#### Usage

```bash
featurecraft wizard [OPTIONS]
```

#### Optional Arguments

```bash
--output PATH, -o PATH              Output config file path
--format {yaml,json,toml}           Output format (default: yaml)
```

#### Examples

```bash
# Interactive wizard
featurecraft wizard --output my_config.yaml

# Wizard with JSON output
featurecraft wizard --output config.json --format json
```

#### Interactive Prompts

The wizard will ask:
1. **Random seed**: Reproducibility
2. **Estimator family**: Model type (tree, linear, etc.)
3. **Cardinality thresholds**: Encoding strategy
4. **Class imbalance**: SMOTE settings
5. **Missing values**: Imputation strategy
6. **Drift detection**: Enable monitoring
7. **Advanced features**: SHAP, schema validation

#### Output

- `{output}`: Generated configuration file
- Console: Configuration summary

---

## Configuration

### Config File Formats

FeatureCraft supports multiple config formats:

#### YAML (Recommended)

```yaml
# config.yaml
random_state: 42
artifacts_dir: "my_artifacts"

# Encoding
low_cardinality_max: 15
use_target_encoding: true

# Imbalance
use_smote: true
smote_threshold: 0.10

# Drift
enable_drift_detection: true
drift_psi_threshold: 0.25
```

#### JSON

```json
{
  "random_state": 42,
  "artifacts_dir": "my_artifacts",
  "low_cardinality_max": 15,
  "use_target_encoding": true,
  "use_smote": true,
  "smote_threshold": 0.10
}
```

#### TOML

```toml
[config]
random_state = 42
artifacts_dir = "my_artifacts"
low_cardinality_max = 15
use_target_encoding = true
use_smote = true
smote_threshold = 0.10
```

### Configuration Precedence

Configuration is merged in this order (highest to lowest priority):

1. **CLI `--set` overrides** (highest)
2. **Explicit config file** (`--config`)
3. **Environment variables** (`FEATURECRAFT__*`)
4. **Library defaults** (lowest)

### Environment Variables

```bash
# Set via environment
export FEATURECRAFT__RANDOM_STATE=42
export FEATURECRAFT__LOW_CARDINALITY_MAX=15
export FEATURECRAFT__USE_SMOTE=true
export FEATURECRAFT__SMOTE_THRESHOLD=0.10

# Run command (env vars applied)
featurecraft fit --input train.csv --target target
```

### CLI Overrides

```bash
# Override specific parameters
featurecraft fit \
  --input train.csv \
  --target target \
  --config base.yaml \
  --set low_cardinality_max=20 \
  --set use_smote=true \
  --set smote_threshold=0.15 \
  --random-state 999
```

---

## Examples

### Complete Workflow

```bash
# 1. Analyze dataset
featurecraft analyze \
  --input data.csv \
  --target survived \
  --out analysis/ \
  --open

# 2. Generate config
featurecraft wizard --output titanic_config.yaml

# 3. Fit pipeline
featurecraft fit \
  --input train.csv \
  --target survived \
  --config titanic_config.yaml \
  --estimator-family tree \
  --out pipeline/

# 4. Transform test data
featurecraft transform \
  --input test.csv \
  --pipeline pipeline/pipeline.joblib \
  --output test_transformed.csv
```

### Production Pipeline

```bash
# Training phase
featurecraft fit-transform \
  --input train.csv \
  --target target \
  --config prod_config.yaml \
  --estimator-family tree \
  --output-data train_transformed.csv \
  --out prod_pipeline/

# Inference phase
featurecraft transform \
  --input new_data.csv \
  --pipeline prod_pipeline/pipeline.joblib \
  --output predictions_input.csv
```

### With Drift Detection

```bash
# Weekly drift monitoring
featurecraft analyze \
  --input current_week.csv \
  --target target \
  --reference training_data.csv \
  --set enable_drift_detection=true \
  --set drift_psi_threshold=0.25 \
  --out drift_reports/week_$(date +%Y%m%d)/
```

### Hyperparameter Search

```bash
# Test multiple configs
for low_card in 10 15 20; do
  for use_smote in true false; do
    featurecraft fit-transform \
      --input train.csv \
      --target target \
      --set low_cardinality_max=$low_card \
      --set use_smote=$use_smote \
      --output-data "train_${low_card}_${use_smote}.csv" \
      --out "pipeline_${low_card}_${use_smote}/"
  done
done
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | File not found |
| 4 | Configuration error |
| 5 | Data validation error |

---

## Shell Completion

### Bash

```bash
# Add to ~/.bashrc
eval "$(_FEATURECRAFT_COMPLETE=bash_source featurecraft)"
```

### Zsh

```bash
# Add to ~/.zshrc
eval "$(_FEATURECRAFT_COMPLETE=zsh_source featurecraft)"
```

### Fish

```bash
# Add to ~/.config/fish/config.fish
eval (env _FEATURECRAFT_COMPLETE=fish_source featurecraft)
```

---

## Best Practices

1. **Use config files for reproducibility**: Version control your configs
2. **Start with `analyze`**: Understand your data before fitting
3. **Test configs with `print-config`**: Verify merged configuration
4. **Enable drift detection in production**: Monitor data quality
5. **Use `--dry-run` for validation**: Test without writing files (if supported)
6. **Save transformed data**: Keep audit trail of transformations
7. **Version pipelines**: Tag pipeline exports with dates/versions

---

## Troubleshooting

See [Troubleshooting Guide](troubleshooting.md) for common CLI errors and solutions.

---

## See Also

- [API Reference](api-reference.md) - Python API
- [Configuration Guide](configuration.md) - All parameters
- [Getting Started](getting-started.md) - Quick start
- [Advanced Features](advanced-features.md) - Drift, leakage, SHAP

