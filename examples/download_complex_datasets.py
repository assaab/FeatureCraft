"""
Helper script to download COMPLEX Kaggle datasets for advanced benchmarking.

Usage:
    python download_complex_datasets.py --all
    python download_complex_datasets.py --datasets ieee_fraud,santander,home_credit

Required Kaggle datasets:
1. IEEE-CIS Fraud Detection (ieee-fraud-detection)
2. Santander Customer Transaction (santander-customer-transaction-prediction)
3. Home Credit Default Risk (home-credit-default-risk)

Note: These are large datasets (several GB). Ensure you have enough disk space.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.panel import Panel
    from rich import box
except ImportError:
    print("Installing rich...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "rich"])
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.panel import Panel
    from rich import box

console = Console()


COMPLEX_DATASETS = {
    'ieee_fraud': {
        'type': 'competition',
        'name': 'ieee-fraud-detection',
        'dir': 'ieee-fraud-detection',
        'files': ['train_transaction.csv', 'train_identity.csv'],
        'description': 'IEEE-CIS Fraud Detection',
        'size': '~650 MB',
        'challenges': [
            '434 features (transaction + identity)',
            'High cardinality categoricals',
            'Heavy class imbalance (3.5%)',
            '50%+ missing data'
        ]
    },
    'santander': {
        'type': 'competition',
        'name': 'santander-customer-transaction-prediction',
        'dir': 'santander-transaction',
        'files': ['train.csv'],
        'description': 'Santander Customer Transaction Prediction',
        'size': '~100 MB',
        'challenges': [
            '200 anonymized numeric features',
            'Class imbalance (10%)',
            'Distribution shifts',
            'Pure numeric optimization'
        ]
    },
    'home_credit': {
        'type': 'competition',
        'name': 'home-credit-default-risk',
        'dir': 'home-credit-default-risk',
        'files': ['application_train.csv', 'bureau.csv', 'previous_application.csv'],
        'description': 'Home Credit Default Risk',
        'size': '~200 MB',
        'challenges': [
            'Multi-table relational (7 tables)',
            'Feature aggregation required',
            '122+ features',
            'Temporal patterns'
        ]
    }
}


def check_kaggle_api():
    """Check if Kaggle API is installed and configured."""
    try:
        import kaggle
        console.print("[green]✓[/green] Kaggle API is installed")
        
        # Test credentials
        try:
            kaggle.api.authenticate()
            console.print("[green]✓[/green] Kaggle API credentials valid")
            return True
        except Exception as e:
            console.print(f"[red]✗[/red] Kaggle API authentication failed: {e}")
            return False
            
    except ImportError:
        console.print("[red]✗[/red] Kaggle API not installed")
        console.print("  Install with: [cyan]pip install kaggle[/cyan]")
        return False
    except OSError as e:
        console.print("[red]✗[/red] Kaggle API credentials not found")
        console.print("\n[bold yellow]Setup Instructions:[/bold yellow]")
        console.print("  1. Go to [cyan]https://www.kaggle.com/account[/cyan]")
        console.print("  2. Click '[bold]Create New API Token[/bold]'")
        console.print("  3. Place [cyan]kaggle.json[/cyan] in:")
        if sys.platform == "win32":
            console.print("     [cyan]C:\\Users\\<username>\\.kaggle\\kaggle.json[/cyan]")
        else:
            console.print("     [cyan]~/.kaggle/kaggle.json[/cyan]")
            console.print("  4. Run: [cyan]chmod 600 ~/.kaggle/kaggle.json[/cyan]")
        return False


def download_competition(name: str, output_dir: Path):
    """Download a Kaggle competition dataset."""
    try:
        import kaggle
        console.print(f"  [cyan]Downloading competition: {name}...[/cyan]")
        kaggle.api.competition_download_files(name, path=str(output_dir))
        return True
    except Exception as e:
        console.print(f"  [red]Error:[/red] {e}")
        return False


def unzip_files(directory: Path):
    """Unzip all zip files in a directory."""
    import zipfile
    
    zip_files = list(directory.glob("*.zip"))
    if not zip_files:
        return
    
    console.print(f"  [cyan]Extracting {len(zip_files)} file(s)...[/cyan]")
    
    for zip_file in zip_files:
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(directory)
            console.print(f"    [green]✓[/green] Extracted {zip_file.name}")
            # Remove zip file after extraction
            zip_file.unlink()
        except Exception as e:
            console.print(f"    [red]✗[/red] Failed to extract {zip_file.name}: {e}")


def verify_files(directory: Path, expected_files: List[str]) -> bool:
    """Verify that expected files exist."""
    found_files = []
    missing_files = []
    
    for file_name in expected_files:
        file_path = directory / file_name
        if file_path.exists():
            found_files.append(file_name)
        else:
            missing_files.append(file_name)
    
    if missing_files:
        console.print(f"  [yellow]Missing files:[/yellow] {', '.join(missing_files)}")
        console.print(f"  [green]Found files:[/green] {', '.join([f.name for f in directory.iterdir() if f.is_file()][:10])}")
    
    return len(missing_files) == 0


def download_dataset_wrapper(dataset_key: str, base_dir: Path):
    """Download and extract a complex dataset."""
    dataset_info = COMPLEX_DATASETS[dataset_key]
    output_dir = base_dir / dataset_info['dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print dataset info
    console.print(f"\n[bold cyan]{'='*70}[/bold cyan]")
    console.print(f"[bold cyan]📦 {dataset_info['description']}[/bold cyan]")
    console.print(f"[bold cyan]{'='*70}[/bold cyan]")
    console.print(f"[yellow]Size:[/yellow] {dataset_info['size']}")
    console.print(f"[yellow]Challenges:[/yellow]")
    for challenge in dataset_info['challenges']:
        console.print(f"  • {challenge}")
    
    # Check if already exists
    if verify_files(output_dir, dataset_info['files']):
        console.print(f"\n[green]✓[/green] Already downloaded to: [cyan]{output_dir}[/cyan]")
        return True
    
    # Download
    console.print(f"\n[cyan]📥 Downloading from Kaggle...[/cyan]")
    if dataset_info['type'] == 'competition':
        success = download_competition(dataset_info['name'], output_dir)
    else:
        console.print(f"  [red]Unsupported dataset type: {dataset_info['type']}[/red]")
        return False
    
    if not success:
        console.print("  [red]✗[/red] Download failed")
        return False
    
    # Unzip
    unzip_files(output_dir)
    
    # Verify
    if verify_files(output_dir, dataset_info['files']):
        console.print(f"\n[green]✓[/green] Successfully downloaded to: [cyan]{output_dir}[/cyan]")
        return True
    else:
        console.print(f"\n[yellow]⚠[/yellow] Downloaded but some expected files are missing")
        console.print(f"  [yellow]The benchmark will work but may have limited data[/yellow]")
        return True  # Return True anyway since we have some files


def main():
    parser = argparse.ArgumentParser(
        description="Download Complex Kaggle Datasets for Advanced Benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_complex_datasets.py --all
  python download_complex_datasets.py --datasets ieee_fraud,santander
  python download_complex_datasets.py --datasets home_credit --data-dir ./my_data

Note: These are competition datasets. You must:
  1. Accept the competition rules on Kaggle.com
  2. Have a valid Kaggle API token configured
        """
    )
    parser.add_argument(
        '--datasets',
        type=str,
        default='all',
        help='Comma-separated dataset names or "all" (ieee_fraud, santander, home_credit)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Directory to store datasets (default: ./data)'
    )
    args = parser.parse_args()
    
    # Header
    console.print()
    console.print(Panel.fit(
        "[bold red]Complex Kaggle Dataset Downloader[/bold red]\n"
        "[yellow]For Advanced FeatureCraft Benchmarking[/yellow]\n"
        "[cyan]IEEE Fraud • Home Credit • Santander[/cyan]",
        border_style="red",
        box=box.DOUBLE
    ))
    console.print()
    
    # Check Kaggle API
    if not check_kaggle_api():
        console.print("\n[bold red]❌ Please install and configure Kaggle API first[/bold red]")
        console.print("\n[yellow]Quick Start:[/yellow]")
        console.print("  1. [cyan]pip install kaggle[/cyan]")
        console.print("  2. Download your API token from kaggle.com/account")
        console.print("  3. Place it in the .kaggle directory")
        sys.exit(1)
    
    # Determine which datasets to download
    if args.datasets.lower() == 'all':
        selected = list(COMPLEX_DATASETS.keys())
    else:
        selected = [d.strip() for d in args.datasets.split(',')]
        # Validate
        invalid = [d for d in selected if d not in COMPLEX_DATASETS]
        if invalid:
            console.print(f"\n[red]❌ Invalid datasets:[/red] {', '.join(invalid)}")
            console.print(f"[yellow]Available:[/yellow] {', '.join(COMPLEX_DATASETS.keys())}")
            sys.exit(1)
    
    console.print(f"[cyan]📊 Datasets to download:[/cyan] {', '.join(selected)}")
    console.print(f"[cyan]📁 Output directory:[/cyan] {args.data_dir}\n")
    
    # Calculate total size
    total_size_info = ' + '.join([COMPLEX_DATASETS[k]['size'] for k in selected])
    console.print(f"[yellow]⚠️  Total size:[/yellow] {total_size_info}")
    console.print(f"[yellow]⚠️  Ensure you have sufficient disk space and accept competition rules![/yellow]\n")
    
    base_dir = Path(args.data_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Download datasets
    results = {}
    for dataset_key in selected:
        success = download_dataset_wrapper(dataset_key, base_dir)
        results[dataset_key] = success
    
    # Summary
    console.print(f"\n[bold blue]{'='*70}[/bold blue]")
    console.print(f"[bold blue]📋 Download Summary[/bold blue]")
    console.print(f"[bold blue]{'='*70}[/bold blue]\n")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    for dataset_key, success in results.items():
        status = "[green]✓[/green]" if success else "[red]✗[/red]"
        console.print(f"{status} {COMPLEX_DATASETS[dataset_key]['description']}")
    
    console.print(f"\n[bold]Result:[/bold] {success_count}/{total_count} datasets")
    
    if success_count == total_count:
        console.print("\n[bold green]🎉 All datasets downloaded successfully![/bold green]")
        console.print("\n[bold]Next steps:[/bold]")
        console.print("  Run the complex benchmark:")
        console.print("  [cyan]python examples/03_complex_kaggle_benchmark.py --datasets all[/cyan]")
    elif success_count > 0:
        console.print("\n[bold yellow]⚠️  Some downloads succeeded[/bold yellow]")
        console.print("You can run the benchmark with downloaded datasets:")
        console.print(f"  [cyan]python examples/03_complex_kaggle_benchmark.py --datasets {','.join([k for k, v in results.items() if v])}[/cyan]")
    else:
        console.print("\n[bold red]❌ All downloads failed[/bold red]")
        console.print("The benchmark will use synthetic fallback data.")
        console.print("\n[yellow]Troubleshooting:[/yellow]")
        console.print("  • Ensure you've accepted the competition rules on Kaggle")
        console.print("  • Verify your Kaggle API credentials")
        console.print("  • Check your internet connection")
    
    console.print()


if __name__ == "__main__":
    main()


