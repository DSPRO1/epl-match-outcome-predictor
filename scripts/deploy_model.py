"""
Deploy trained model to Modal.

This script uploads the trained model to Modal's volume storage
for use by the inference API.

Usage:
    python scripts/deploy_model.py [--model MODEL_NAME]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import modal
from src.config import (
    RANDOM_FOREST_MODEL, XGBOOST_MODEL, LIGHTGBM_MODEL,
    FEATURES_FILE, MODAL_APP_NAME
)


def main():
    parser = argparse.ArgumentParser(description='Deploy model to Modal')
    parser.add_argument(
        '--model',
        choices=['random_forest', 'xgboost', 'lightgbm'],
        default='random_forest',
        help='Which model to deploy (default: random_forest)'
    )
    args = parser.parse_args()

    # Map model names to files
    model_files = {
        'random_forest': RANDOM_FOREST_MODEL,
        'xgboost': XGBOOST_MODEL,
        'lightgbm': LIGHTGBM_MODEL
    }

    model_path = model_files[args.model]

    print("=" * 60)
    print(f"Deploying {args.model} to Modal")
    print("=" * 60)

    # Check if model exists
    if not model_path.exists():
        print(f"\n✗ Model not found: {model_path}")
        print(f"  Run 'python scripts/train_models.py --model {args.model}' first")
        sys.exit(1)

    if not FEATURES_FILE.exists():
        print(f"\n✗ Features file not found: {FEATURES_FILE}")
        print("  Run 'python scripts/train_models.py' first")
        sys.exit(1)

    # Load model and features
    print("\nLoading model and features...")
    with open(model_path, 'rb') as f:
        model_bytes = f.read()

    with open(FEATURES_FILE, 'rb') as f:
        features_bytes = f.read()

    # Upload to Modal
    print("Uploading to Modal...")
    app = modal.App(MODAL_APP_NAME)

    # Get the upload_model function from modal_api
    upload_fn = modal.Function.lookup(MODAL_APP_NAME, "upload_model")

    result = upload_fn.remote(model_bytes, features_bytes)

    print("\n✓ Deployment successful!")
    print(f"\nModel: {result['model_type']}")
    print(f"Features: {len(result['features'])} features")
    print(f"Path: {result['model_path']}")

    print("\n" + "=" * 60)
    print("Deployment complete!")
    print("=" * 60)
    print("\nAPI endpoint: https://dspro1--epl-predictor-fastapi-app.modal.run")


if __name__ == "__main__":
    main()
