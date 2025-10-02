"""CLI interface for FeatureCraft with comprehensive configuration support."""

from __future__ import annotations

import json
import os
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from .config import FeatureCraftConfig
from .pipeline import AutoFeatureEngineer
from .report import ReportBuilder
from .settings import config_to_dict, load_config

app = typer.Typer(add_completion=False, help="FeatureCraft: Automatic Feature Engineering")
console = Console()


@app.command()
def wizard(output: str = typer.Option("featurecraft-config.yaml", help="Output config file path")):
    """Launch interactive configuration wizard to generate a config file."""
    from .interactive import run_wizard

    try:
        cfg = run_wizard(output_path=output)
        console.print(f"\n[green]✓ Configuration wizard complete![/green]")
        console.print(f"Config saved to: {output}")
        console.print(f"\nUse with: featurecraft fit --config {output} ...")
    except KeyboardInterrupt:
        console.print("\n[yellow]Wizard cancelled.[/yellow]")
        raise typer.Exit(0)
    except Exception as e:
        console.print(f"[red]Wizard failed: {e}[/red]")
        raise typer.Exit(1)


@app.command("print-config")
def print_config(
    config: Optional[str] = typer.Option(None, help="Config file to load (YAML/JSON/TOML)"),
    set_flags: Optional[List[str]] = typer.Option(None, "--set", help="Config overrides (key=value)"),
    format: str = typer.Option("yaml", help="Output format: yaml, json, or toml"),
    schema: bool = typer.Option(False, help="Print JSON schema instead of config values"),
    flatten: bool = typer.Option(False, help="Flatten nested keys (e.g., reducer.kind)"),
):
    """Print effective configuration after merging all sources, or export schema."""
    if schema:
        # Export JSON Schema
        schema_dict = FeatureCraftConfig.model_json_schema()
        console.print(json.dumps(schema_dict, indent=2))
        return

    try:
        cfg = load_config(config_file=config, cli_overrides=set_flags or [], use_env=True)
        config_dict = config_to_dict(cfg, flatten=flatten)

        if format == "yaml":
            import yaml

            console.print(yaml.dump(config_dict, sort_keys=False, indent=2))
        elif format == "json":
            console.print(json.dumps(config_dict, indent=2))
        elif format == "toml":
            try:
                import tomli_w

                console.print(tomli_w.dumps(config_dict))
            except ImportError:
                console.print("[red]TOML export requires: pip install tomli-w[/red]")
                raise typer.Exit(1)
        else:
            console.print(f"[red]Unsupported format: {format}[/red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Failed to load/print config: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def analyze(
    input: str = typer.Option(..., help="Input CSV/Parquet file"),
    target: str = typer.Option(..., help="Target column name"),
    out: str = typer.Option("artifacts", help="Artifacts output directory"),
    config: Optional[str] = typer.Option(None, help="Config file path (YAML/JSON/TOML)"),
    set_flags: Optional[List[str]] = typer.Option(None, "--set", help="Config overrides (key=value)"),
    random_state: Optional[int] = typer.Option(None, help="Random seed override"),
    interactive: bool = typer.Option(False, "--interactive/--no-interactive", help="Enable interactive prompts"),
):
    """Analyze dataset and generate insights report."""
    try:
        # Load configuration
        api_kwargs = {"artifacts_dir": out}
        if random_state is not None:
            api_kwargs["random_state"] = random_state

        cfg = load_config(
            config_file=config, cli_overrides=set_flags or [], api_kwargs=api_kwargs, use_env=True
        )

        # Read data
        df = _read_frame(input)

        # Create AFE with config
        afe = AutoFeatureEngineer(config=cfg)
        insights = afe.analyze(df, target=target)

        # Print issues table
        if insights.issues:
            table = Table(title="FeatureCraft Issues")
            table.add_column("Severity")
            table.add_column("Code")
            table.add_column("Column")
            table.add_column("Message")
            for issue in insights.issues:
                table.add_row(issue.severity, issue.code or "-", issue.column or "-", issue.message)
            console.print(table)

        # Save report
        os.makedirs(out, exist_ok=True)

        # Get templates directory
        templates_dir = _get_templates_dir()

        rb = ReportBuilder(templates_dir=templates_dir)
        report_path = os.path.join(out, cfg.report_filename)
        rb.build(insights, report_path, cfg)

        console.print(f"[green]✓ Report saved to {report_path}[/green]")

        # Open in browser if configured
        if cfg.open_report:
            import webbrowser

            webbrowser.open(f"file://{os.path.abspath(report_path)}")

    except FileNotFoundError as e:
        console.print(f"[red]File not found: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Analysis failed: {e}[/red]")
        raise typer.Exit(1)


@app.command("fit")
def fit(
    input: str = typer.Option(..., help="Input CSV/Parquet file"),
    target: str = typer.Option(..., help="Target column name"),
    out: str = typer.Option("artifacts", help="Artifacts output directory"),
    estimator_family: str = typer.Option("tree", help="Estimator family: tree, linear, svm, knn, nn"),
    config: Optional[str] = typer.Option(None, help="Config file path (YAML/JSON/TOML)"),
    set_flags: Optional[List[str]] = typer.Option(None, "--set", help="Config overrides (key=value)"),
    random_state: Optional[int] = typer.Option(None, help="Random seed override"),
    interactive: bool = typer.Option(False, "--interactive/--no-interactive", help="Enable interactive prompts"),
    dry_run: bool = typer.Option(False, "--dry-run/--no-dry-run", help="Dry run (no file writes)"),
):
    """Fit feature engineering pipeline and export artifacts."""
    try:
        # Load configuration
        api_kwargs = {"artifacts_dir": out, "dry_run": dry_run}
        if random_state is not None:
            api_kwargs["random_state"] = random_state

        cfg = load_config(
            config_file=config, cli_overrides=set_flags or [], api_kwargs=api_kwargs, use_env=True
        )

        # Read data
        df = _read_frame(input)
        X = df.drop(columns=[target])
        y = df[target]

        # Create AFE with config
        afe = AutoFeatureEngineer(config=cfg)

        # Interactive questions if enabled
        if interactive:
            from .interactive import interactive_fit_questions

            console.print("[cyan]Interactive mode enabled. Analyzing data...[/cyan]")
            overrides = interactive_fit_questions(X, y, cfg)
            if overrides:
                console.print(f"[green]Applying {len(overrides)} interactive overrides[/green]")
                afe.set_params(**overrides)

        # Fit pipeline
        console.print(f"[cyan]Fitting pipeline for estimator family: {estimator_family}[/cyan]")
        afe.fit(X, y, estimator_family=estimator_family)

        # Export artifacts
        if not cfg.dry_run:
            afe.export(out)
            console.print(f"[green]✓ Pipeline exported to {out}[/green]")
        else:
            console.print(f"[yellow]Dry run: would export to {out}[/yellow]")

    except FileNotFoundError as e:
        console.print(f"[red]File not found: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Fit failed: {e}[/red]")
        if cfg.fail_fast:
            raise
        raise typer.Exit(1)


@app.command("transform")
def transform(
    input: str = typer.Option(..., help="Input CSV/Parquet file"),
    target: str = typer.Option(
        None, help="Target column name (will be dropped if present, optional for inference data)"
    ),
    pipeline_dir: str = typer.Option("artifacts", help="Directory containing pipeline.joblib"),
    output: str = typer.Option("artifacts/transformed.parquet", help="Output file path (.parquet)"),
):
    """Transform data using fitted pipeline."""
    try:
        # Read data
        df = _read_frame(input)
        if target and target in df.columns:
            X = df.drop(columns=[target])
        else:
            X = df

        # Load pipeline
        pipeline_path = os.path.join(pipeline_dir, "pipeline.joblib")
        if not os.path.exists(pipeline_path):
            console.print(f"[red]Pipeline not found at {pipeline_path}[/red]")
            console.print(f"[yellow]Hint: Run 'featurecraft fit' first[/yellow]")
            raise typer.Exit(1)

        pipeline = joblib.load(pipeline_path)

        # Transform
        console.print("[cyan]Transforming data...[/cyan]")
        Xt = pipeline.transform(X)

        # Convert to dense if sparse
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()

        # Load feature names
        fn_path = os.path.join(pipeline_dir, "feature_names.txt")
        if os.path.exists(fn_path):
            with open(fn_path, encoding="utf-8") as f:
                names = [line.strip() for line in f]
        else:
            names = [f"f_{i}" for i in range(Xt.shape[1])]

        # Create DataFrame
        out_df = pd.DataFrame(np.asarray(Xt), columns=names)

        # Save output
        out_dir = os.path.dirname(output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out_df.to_parquet(output, index=False)

        console.print(f"[green]✓ Transformed data saved to {output}[/green]")
        console.print(f"   Shape: {out_df.shape}")

    except FileNotFoundError as e:
        console.print(f"[red]File not found: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Transform failed: {e}[/red]")
        raise typer.Exit(1)


@app.command("fit-transform")
def fit_transform(
    input: str = typer.Option(..., help="Input CSV/Parquet file"),
    target: str = typer.Option(..., help="Target column name"),
    out: str = typer.Option("artifacts", help="Artifacts output directory"),
    estimator_family: str = typer.Option("tree", help="Estimator family: tree, linear, svm, knn, nn"),
    output: str = typer.Option("artifacts/transformed.parquet", help="Output transformed data path"),
    config: Optional[str] = typer.Option(None, help="Config file path (YAML/JSON/TOML)"),
    set_flags: Optional[List[str]] = typer.Option(None, "--set", help="Config overrides (key=value)"),
    random_state: Optional[int] = typer.Option(None, help="Random seed override"),
    interactive: bool = typer.Option(False, "--interactive/--no-interactive", help="Enable interactive prompts"),
    dry_run: bool = typer.Option(False, "--dry-run/--no-dry-run", help="Dry run (no file writes)"),
):
    """Fit pipeline and transform data in one step."""
    try:
        # Load configuration
        api_kwargs = {"artifacts_dir": out, "dry_run": dry_run}
        if random_state is not None:
            api_kwargs["random_state"] = random_state

        cfg = load_config(
            config_file=config, cli_overrides=set_flags or [], api_kwargs=api_kwargs, use_env=True
        )

        # Read data
        df = _read_frame(input)
        X = df.drop(columns=[target])
        y = df[target]

        # Create AFE with config
        afe = AutoFeatureEngineer(config=cfg)

        # Interactive questions if enabled
        if interactive:
            from .interactive import interactive_fit_questions

            console.print("[cyan]Interactive mode enabled. Analyzing data...[/cyan]")
            overrides = interactive_fit_questions(X, y, cfg)
            if overrides:
                console.print(f"[green]Applying {len(overrides)} interactive overrides[/green]")
                afe.set_params(**overrides)

        # Fit and transform
        console.print(f"[cyan]Fitting pipeline for estimator family: {estimator_family}[/cyan]")
        Xt = afe.fit_transform(X, y, estimator_family=estimator_family)

        # Export artifacts
        if not cfg.dry_run:
            afe.export(out)
            console.print(f"[green]✓ Pipeline exported to {out}[/green]")

            # Save transformed data
            out_dir = os.path.dirname(output)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            Xt.to_parquet(output, index=False)
            console.print(f"[green]✓ Transformed data saved to {output}[/green]")
            console.print(f"   Shape: {Xt.shape}")
        else:
            console.print(f"[yellow]Dry run: would export to {out} and {output}[/yellow]")
            console.print(f"   Would transform to shape: {Xt.shape}")

    except FileNotFoundError as e:
        console.print(f"[red]File not found: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Fit-transform failed: {e}[/red]")
        if cfg.fail_fast:
            raise
        raise typer.Exit(1)


def _read_frame(path: str) -> pd.DataFrame:
    """Read DataFrame from CSV or Parquet file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    try:
        if path.lower().endswith(".csv"):
            return pd.read_csv(path)
        if path.lower().endswith(".parquet"):
            return pd.read_parquet(path)
        raise ValueError("Unsupported file type. Use .csv or .parquet")
    except Exception as e:
        console.print(f"[red]Error reading file {path}: {e}[/red]")
        raise


def _get_templates_dir() -> str:
    """Get templates directory path, handling both dev and installed package."""
    try:
        # Try Python 3.9+ importlib.resources.files API
        import importlib.resources as pkg_resources

        try:
            templates_ref = pkg_resources.files("featurecraft").joinpath("templates")
            templates_dir = str(templates_ref)
        except (TypeError, AttributeError):
            # Fallback for older API or edge cases
            templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    except Exception:
        # Ultimate fallback: relative to this file
        templates_dir = os.path.join(os.path.dirname(__file__), "templates")

    # Verify templates directory exists
    if not os.path.exists(templates_dir):
        # Try alternative location (for editable installs)
        alt_templates = os.path.join(os.path.dirname(__file__), "..", "..", "templates")
        if os.path.exists(alt_templates):
            templates_dir = alt_templates
        else:
            console.print(f"[red]Error: Templates directory not found at {templates_dir}[/red]")
            console.print("[yellow]Hint: Install package properly or check installation[/yellow]")
            raise FileNotFoundError(f"Templates not found at {templates_dir}")

    return templates_dir


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()

