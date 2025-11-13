"""
W&B Configuration for EPL Match Outcome Predictor
This module handles all Weights & Biases initialization and configuration.
"""

import os
import wandb

# W&B API Key
WANDB_API_KEY = "6a59e9eeb33878d0ad8828b5b134749f96ff1f82"

# W&B Project Configuration
WANDB_ENTITY = "philip-baumann-hslu"
WANDB_PROJECT = "epl-match-outcome-predictor"

def setup_wandb():
    """
    Initialize W&B with API key and return True if successful.
    """
    os.environ['WANDB_API_KEY'] = WANDB_API_KEY
    return True

def init_wandb_run(model_name, config_params):
    """
    Initialize a new W&B run for a specific model.

    Args:
        model_name: Name of the model (e.g., 'RandomForest', 'XGBoost', 'LightGBM')
        config_params: Dictionary of hyperparameters and configuration

    Returns:
        wandb.Run object
    """
    setup_wandb()

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=f"{model_name}-run",
        config=config_params,
        reinit=True  # Allow multiple runs in same script
    )

    return run
