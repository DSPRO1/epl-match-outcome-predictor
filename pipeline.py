"""
EPL Match Outcome Predictor - Full ML Pipeline

This orchestrator script runs the complete pipeline:
1. Download/update match data
2. Train models with W&B tracking
3. Update database with team statistics
4. (Optional) Deploy model to Modal

Usage:
    # Run full pipeline
    python pipeline.py

    # Use existing data (skip download)
    python pipeline.py --skip-download

    # Run specific steps
    python pipeline.py --steps download train

    # Skip database update
    python pipeline.py --skip-db

    # Train only specific model
    python pipeline.py --model random_forest

    # Deploy after training
    python pipeline.py --deploy
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(script_path: str, args: list = None):
    """
    Run a pipeline step script.

    Args:
        script_path: Path to the script
        args: Optional list of arguments

    Returns:
        True if successful, False otherwise
    """
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)

    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print('='*80)

    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description='EPL Match Outcome Predictor - Full Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--steps',
        nargs='+',
        choices=['download', 'train', 'database', 'deploy'],
        help='Run specific pipeline steps (default: all except deploy)'
    )

    parser.add_argument(
        '--model',
        choices=['random_forest', 'xgboost', 'lightgbm', 'all'],
        default='all',
        help='Which model to train (default: all)'
    )

    parser.add_argument(
        '--skip-db',
        action='store_true',
        help='Skip database update'
    )

    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip data download (use existing data/matches.csv)'
    )

    parser.add_argument(
        '--deploy',
        action='store_true',
        help='Deploy model to Modal after training'
    )

    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable Weights & Biases tracking'
    )

    args = parser.parse_args()

    # Determine which steps to run
    if args.steps:
        steps = args.steps
    else:
        steps = ['download', 'train', 'database']
        if args.deploy:
            steps.append('deploy')

    if args.skip_db and 'database' in steps:
        steps.remove('database')

    if args.skip_download and 'download' in steps:
        steps.remove('download')

    print("=" * 80)
    print("EPL MATCH OUTCOME PREDICTOR - ML PIPELINE")
    print("=" * 80)
    print(f"\nSteps to run: {', '.join(steps)}")
    print(f"Model: {args.model}")
    if args.no_wandb:
        print("W&B tracking: Disabled")
    if args.skip_download:
        print("Data download: Skipped (using existing data)")
    print()

    success = True

    # Step 1: Download data
    if 'download' in steps:
        print("\n[STEP 1/4] Downloading match data...")
        if not run_step('scripts/download_data.py'):
            print("✗ Data download failed")
            sys.exit(1)

    # Step 2: Train models
    if 'train' in steps:
        print("\n[STEP 2/4] Training models...")
        train_args = ['--model', args.model]
        if args.no_wandb:
            train_args.append('--no-wandb')

        if not run_step('scripts/train_models.py', train_args):
            print("✗ Model training failed")
            sys.exit(1)

    # Step 3: Update database
    if 'database' in steps:
        print("\n[STEP 3/4] Updating database...")
        if not run_step('scripts/update_database.py'):
            print("✗ Database update failed")
            sys.exit(1)

    # Step 4: Deploy (optional)
    if 'deploy' in steps:
        print("\n[STEP 4/4] Deploying to Modal...")
        deploy_model = args.model if args.model != 'all' else 'random_forest'
        if not run_step('scripts/deploy_model.py', ['--model', deploy_model]):
            print("✗ Deployment failed")
            sys.exit(1)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)

    print("\nNext steps:")
    if 'deploy' not in steps:
        print("  • Deploy model: python scripts/deploy_model.py")
    print("  • Start web UI: cd web-ui && bun run dev")
    print("  • View W&B dashboard: https://wandb.ai/philip-baumann-hslu/epl-match-outcome-predictor")


if __name__ == "__main__":
    main()
