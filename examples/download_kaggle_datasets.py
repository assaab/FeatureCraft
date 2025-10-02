"""
Helper script to download Kaggle datasets for benchmarking.

Usage:
    python download_kaggle_datasets.py --all
    python download_kaggle_datasets.py --datasets titanic,house_prices
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
except ImportError:
    print("Installing rich...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "rich"])
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


DATASETS = {
    'titanic': {
        'type': 'competition',
        'name': 'titanic',
        'dir': 'titanic',
        'files': ['train.csv', 'test.csv'],
        'description': 'Titanic: Machine Learning from Disaster'
    },
    'house_prices': {
        'type': 'competition',
        'name': 'house-prices-advanced-regression-techniques',
        'dir': 'house-prices',
        'files': ['train.csv', 'test.csv'],
        'description': 'House Prices: Advanced Regression Techniques'
    },
    'bike_sharing': {
        'type': 'competition',
        'name': 'bike-sharing-demand',
        'dir': 'bike-sharing',
        'files': ['train.csv', 'test.csv'],
        'description': 'Bike Sharing Demand'
    },
    'telco_churn': {
        'type': 'dataset',
        'name': 'blastchar/telco-customer-churn',
        'dir': 'telco-churn',
        'files': ['WA_Fn-UseC_-Telco-Customer-Churn.csv'],
        'description': 'Telco Customer Churn'
    },
    'credit_card_fraud': {
        'type': 'dataset',
        'name': 'mlg-ulb/creditcardfraud',
        'dir': 'creditcard',
        'files': ['creditcard.csv'],
        'description': 'Credit Card Fraud Detection'
    }
}


def check_kaggle_api():
    """Check if Kaggle API is installed and configured."""
    try:
        import kaggle
        console.print("[green]✓[/green] Kaggle API is installed")
        return True
    except ImportError:
        console.print("[red]✗[/red] Kaggle API not installed")
        console.print("  Install with: pip install kaggle")
        return False
    except OSError as e:
        console.print("[red]✗[/red] Kaggle API credentials not found")
        console.print("  Please setup your kaggle.json file:")
        console.print("  1. Go to https://www.kaggle.com/account")
        console.print("  2. Click 'Create New API Token'")
        console.print("  3. Place kaggle.json in:")
        if sys.platform == "win32":
            console.print("     C:\\Users\\<username>\\.kaggle\\kaggle.json")
        else:
            console.print("     ~/.kaggle/kaggle.json")
            console.print("  4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False


def download_competition(name: str, output_dir: Path):
    """Download a Kaggle competition dataset."""
    try:
        cmd = ['kaggle', 'competitions', 'download', '-c', name, '-p', str(output_dir)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error:[/red] {e.stderr}")
        return False


def download_dataset(name: str, output_dir: Path):
    """Download a Kaggle dataset."""
    try:
        cmd = ['kaggle', 'datasets', 'download', '-d', name, '-p', str(output_dir)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error:[/red] {e.stderr}")
        return False


def unzip_files(directory: Path):
    """Unzip all zip files in a directory."""
    import zipfile
    
    for zip_file in directory.glob("*.zip"):
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(directory)
            console.print(f"  [green]✓[/green] Extracted {zip_file.name}")
            # Remove zip file after extraction
            zip_file.unlink()
        except Exception as e:
            console.print(f"  [red]✗[/red] Failed to extract {zip_file.name}: {e}")


def verify_files(directory: Path, expected_files: List[str]) -> bool:
    """Verify that expected files exist."""
    for file_name in expected_files:
        file_path = directory / file_name
        if not file_path.exists():
            return False
    return True


def download_dataset_wrapper(dataset_key: str, base_dir: Path):
    """Download and extract a dataset."""
    dataset_info = DATASETS[dataset_key]
    output_dir = base_dir / dataset_info['dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    
    # Check if already exists
    if verify_files(output_dir, dataset_info['files']):
        console.print("  [yellow]![/yellow] Already downloaded, skipping")
        return True
    
    # Download
    console.print("  [cyan]Downloading...[/cyan]")
    if dataset_info['type'] == 'competition':
        success = download_competition(dataset_info['name'], output_dir)
    else:
        success = download_dataset(dataset_info['name'], output_dir)
    
    if not success:
        console.print("  [red]✗[/red] Download failed")
        return False
    
    # Unzip
    console.print("  [cyan]Extracting...[/cyan]")
    unzip_files(output_dir)
    
    # Verify
    if verify_files(output_dir, dataset_info['files']):
        console.print("  [green]✓[/green] Successfully downloaded and verified")
        return True
    else:
        console.print("  [red]✗[/red] Some files are missing after extraction")
        console.print(f"     Expected: {', '.join(dataset_info['files'])}")
        console.print(f"     Found: {', '.join([f.name for f in output_dir.iterdir() if f.is_file()])}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download Kaggle datasets for benchmarking")
    parser.add_argument(
        '--datasets',
        type=str,
        default='all',
        help='Comma-separated dataset names or "all" (titanic,house_prices,bike_sharing,telco_churn,credit_card_fraud)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Directory to store datasets (default: ./data)'
    )
    args = parser.parse_args()
    
    # Header
    console.print("\n" + "="*80)
    console.print("[bold blue]Kaggle Dataset Downloader[/bold blue]")
    console.print("="*80 + "\n")
    
    # Check Kaggle API
    if not check_kaggle_api():
        console.print("\n[red]Please install and configure Kaggle API first[/red]")
        sys.exit(1)
    
    # Determine which datasets to download
    if args.datasets == 'all':
        selected = list(DATASETS.keys())
    else:
        selected = [d.strip() for d in args.datasets.split(',')]
        # Validate
        invalid = [d for d in selected if d not in DATASETS]
        if invalid:
            console.print(f"[red]Invalid datasets:[/red] {', '.join(invalid)}")
            console.print(f"[yellow]Available:[/yellow] {', '.join(DATASETS.keys())}")
            sys.exit(1)
    
    console.print(f"[cyan]Datasets to download:[/cyan] {', '.join(selected)}")
    console.print(f"[cyan]Output directory:[/cyan] {args.data_dir}\n")
    
    base_dir = Path(args.data_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Download datasets
    results = {}
    for dataset_key in selected:
        success = download_dataset_wrapper(dataset_key, base_dir)
        results[dataset_key] = success
    
    # Summary
    console.print("\n" + "="*80)
    console.print("[bold blue]Download Summary[/bold blue]")
    console.print("="*80 + "\n")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    for dataset_key, success in results.items():
        status = "[green]✓[/green]" if success else "[red]✗[/red]"
        console.print(f"{status} {DATASETS[dataset_key]['description']}")
    
    console.print(f"\n[bold]Success:[/bold] {success_count}/{total_count} datasets")
    
    if success_count == total_count:
        console.print("\n[bold green]All datasets downloaded successfully![/bold green]")
        console.print("\nYou can now run the benchmark:")
        console.print("  python examples/02_kaggle_benchmark.py --datasets all")
    else:
        console.print("\n[bold yellow]Some downloads failed[/bold yellow]")
        console.print("The benchmark will use synthetic fallback data for missing datasets.")
    
    console.print()


if __name__ == "__main__":
    main()

