"""Configuration loading, merging, and validation for FeatureCraft.

Implements layered configuration system with precedence:
CLI --set overrides > config file > env vars > API kwargs > defaults
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .config import FeatureCraftConfig
from .logging import get_logger

logger = get_logger(__name__)

# Try importing TOML support
try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # Python <3.11
    except ImportError:
        tomllib = None  # type: ignore


def load_from_env(prefix: str = "FEATURECRAFT") -> Dict[str, Any]:
    """Load configuration from environment variables.
    
    Supports nested keys via double underscore: FEATURECRAFT__ENCODERS__LOW_CARDINALITY_MAX=15
    
    Args:
        prefix: Environment variable prefix (default: FEATURECRAFT)
        
    Returns:
        Dict with configuration values
    """
    config = {}
    prefix_with_sep = f"{prefix}__"
    
    for key, value in os.environ.items():
        if not key.startswith(prefix_with_sep):
            continue
        
        # Remove prefix and split by __
        config_key = key[len(prefix_with_sep):].lower()
        parts = config_key.split("__")
        
        # Navigate/create nested structure
        current = config
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Set value with type inference
        current[parts[-1]] = _infer_type(value)
    
    return config


MAX_ENV_VALUE_LENGTH = 10000  # Prevent DoS via huge env vars


def _infer_type(value: str) -> Any:
    """Infer Python type from string value with security checks.
    
    Args:
        value: String value from environment variable
        
    Returns:
        Inferred Python type (bool, int, float, dict, list, or str)
        
    Raises:
        ValueError: If value exceeds maximum length
    """
    # Security: Prevent DoS via huge environment variables
    if len(value) > MAX_ENV_VALUE_LENGTH:
        raise ValueError(
            f"Environment variable value too long (>{MAX_ENV_VALUE_LENGTH} chars). "
            f"Received {len(value)} chars."
        )
    
    # Try boolean
    if value.lower() in ("true", "yes", "1", "on"):
        return True
    if value.lower() in ("false", "no", "0", "off"):
        return False
    
    # Try int
    try:
        return int(value)
    except ValueError:
        pass
    
    # Try float
    try:
        return float(value)
    except ValueError:
        pass
    
    # Try JSON (for lists/dicts)
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Return as string
    return value


def load_from_file(path: str | Path) -> Dict[str, Any]:
    """Load configuration from YAML, TOML, or JSON file.
    
    Args:
        path: Path to configuration file
        
    Returns:
        Dict with configuration values
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is unsupported
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    suffix = path.suffix.lower()
    
    try:
        with open(path, encoding="utf-8") as f:
            if suffix in {".yaml", ".yml"}:
                return yaml.safe_load(f) or {}
            elif suffix == ".json":
                return json.load(f)
            elif suffix == ".toml":
                if tomllib is None:
                    raise ValueError(
                        "TOML support requires tomli package. "
                        "Install with: pip install tomli"
                    )
                # Re-open in binary mode for tomllib
                with open(path, "rb") as fb:
                    return tomllib.load(fb)
            else:
                raise ValueError(f"Unsupported config file format: {suffix}")
    except (OSError, ValueError, yaml.YAMLError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load config from {path}: {e}")
        from .exceptions import ConfigurationError
        raise ConfigurationError(
            f"Failed to load configuration file: {e}",
            config_file=str(path),
            error_type=type(e).__name__
        ) from e


def parse_cli_overrides(overrides: list[str]) -> Dict[str, Any]:
    """Parse CLI --set key=value overrides.
    
    Supports dotted keys for nested settings: encoders.low_cardinality_max=12
    
    Args:
        overrides: List of "key=value" strings
        
    Returns:
        Dict with configuration values
    """
    config = {}
    
    for override in overrides:
        if "=" not in override:
            logger.warning(f"Invalid override format (missing =): {override}")
            continue
        
        key, value = override.split("=", 1)
        key = key.strip()
        value = value.strip()
        
        # Handle dotted keys
        parts = key.split(".")
        current = config
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Set value with type inference
        current[parts[-1]] = _infer_type(value)
    
    return config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge multiple config dicts, with later dicts taking precedence.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration dict
    """
    result = {}
    
    for config in configs:
        result = _deep_merge(result, config)
    
    return result


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dicts, with override taking precedence."""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def flatten_config(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested config dict with dotted keys.
    
    Example: {"encoders": {"max": 10}} -> {"encoders.max": 10}
    """
    flattened = {}
    
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        
        if isinstance(value, dict):
            flattened.update(flatten_config(value, full_key))
        else:
            flattened[full_key] = value
    
    return flattened


def load_config(
    config_file: Optional[str | Path] = None,
    cli_overrides: Optional[list[str]] = None,
    api_kwargs: Optional[Dict[str, Any]] = None,
    use_env: bool = True,
) -> FeatureCraftConfig:
    """Load and merge configuration from all sources.
    
    Precedence order (highest to lowest):
    1. CLI --set overrides
    2. Explicit config file
    3. Environment variables (FEATURECRAFT__*)
    4. Python API kwargs
    5. Library defaults
    
    Args:
        config_file: Path to YAML/TOML/JSON config file
        cli_overrides: List of "key=value" CLI overrides
        api_kwargs: Configuration from Python API
        use_env: Whether to load from environment variables
        
    Returns:
        Validated FeatureCraftConfig instance
    """
    # Start with defaults (implicitly via FeatureCraftConfig)
    merged = {}
    
    # Layer 1: API kwargs
    if api_kwargs:
        merged = _deep_merge(merged, api_kwargs)
        logger.debug(f"Loaded {len(api_kwargs)} settings from API kwargs")
    
    # Layer 2: Environment variables
    if use_env:
        env_config = load_from_env()
        if env_config:
            merged = _deep_merge(merged, env_config)
            logger.debug(f"Loaded {len(flatten_config(env_config))} settings from environment")
    
    # Layer 3: Config file
    if config_file:
        file_config = load_from_file(config_file)
        merged = _deep_merge(merged, file_config)
        logger.info(f"Loaded configuration from {config_file}")
    
    # Layer 4: CLI overrides
    if cli_overrides:
        cli_config = parse_cli_overrides(cli_overrides)
        merged = _deep_merge(merged, cli_config)
        logger.debug(f"Applied {len(cli_overrides)} CLI overrides")
    
    # Validate and return
    try:
        cfg = FeatureCraftConfig(**merged)
        logger.debug("Configuration validated successfully")
        return cfg
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise ValueError(f"Invalid configuration: {e}") from e


def export_schema(output_path: str | Path, format: str = "json") -> None:
    """Export configuration JSON Schema.
    
    Args:
        output_path: Output file path
        format: Output format ("json" or "yaml")
    """
    schema = FeatureCraftConfig.model_json_schema()
    
    # Convert tuples to lists for YAML compatibility
    schema = _convert_tuples_to_lists(schema)
    
    output_path = Path(output_path)
    
    with open(output_path, "w", encoding="utf-8") as f:
        if format == "yaml":
            yaml.safe_dump(schema, f, sort_keys=False, indent=2, default_flow_style=False)
        else:
            json.dump(schema, f, indent=2)
    
    logger.info(f"Exported schema to {output_path}")


def _convert_tuples_to_lists(obj: Any) -> Any:
    """Recursively convert tuples to lists for YAML-safe serialization.
    
    YAML doesn't have a native tuple type, and PyYAML's default behavior
    creates Python-specific tags (!!python/tuple) that fail with safe_load.
    
    Args:
        obj: Object to convert (can be dict, list, tuple, or scalar)
        
    Returns:
        Object with all tuples converted to lists
    """
    if isinstance(obj, dict):
        return {key: _convert_tuples_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_tuples_to_lists(item) for item in obj]
    else:
        return obj


def config_to_dict(cfg: FeatureCraftConfig, flatten: bool = False) -> Dict[str, Any]:
    """Convert FeatureCraftConfig to dict.
    
    Args:
        cfg: Configuration instance
        flatten: Whether to flatten nested keys
        
    Returns:
        Configuration as dict with tuples converted to lists for YAML compatibility
    """
    config_dict = cfg.model_dump()
    
    # Convert tuples to lists for YAML compatibility
    # This ensures yaml.safe_load() can read the file without Python-specific tags
    config_dict = _convert_tuples_to_lists(config_dict)
    
    if flatten:
        return flatten_config(config_dict)
    
    return config_dict


def save_config(cfg: FeatureCraftConfig, path: str | Path, format: str = "yaml") -> None:
    """Save configuration to file.
    
    Args:
        cfg: Configuration instance
        path: Output file path
        format: Output format ("yaml", "json", or "toml")
        
    Note:
        For YAML format, tuples are automatically converted to lists since
        YAML doesn't have a native tuple type. This ensures compatibility
        with yaml.safe_load() when reading the file back.
    """
    path = Path(path)
    config_dict = config_to_dict(cfg)  # Tuples already converted to lists
    
    with open(path, "w", encoding="utf-8") as f:
        if format == "yaml":
            # Use safe_dump for security and compatibility
            # Tuples have already been converted to lists by config_to_dict()
            yaml.safe_dump(config_dict, f, sort_keys=False, indent=2, default_flow_style=False)
        elif format == "json":
            json.dump(config_dict, f, indent=2)
        elif format == "toml":
            # Basic TOML export (requires tomli_w)
            try:
                import tomli_w
                with open(path, "wb") as fb:
                    tomli_w.dump(config_dict, fb)
            except ImportError:
                raise ValueError(
                    "TOML export requires tomli-w package. "
                    "Install with: pip install tomli-w"
                )
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Saved configuration to {path}")

