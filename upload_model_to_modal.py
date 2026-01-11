"""Upload trained model to Modal volume."""

import modal
import pickle
import sys
from pathlib import Path

print("Loading model and features...")

with open("models/random_forest_model.pkl", "rb") as f:
    model_bytes = f.read()

with open("models/features.pkl", "rb") as f:
    features_bytes = f.read()

print(f"Model size: {len(model_bytes):,} bytes")
print(f"Features size: {len(features_bytes)} bytes")

# Create a minimal app for uploading
app = modal.App("epl-predictor-upload")
volume = modal.Volume.from_name("epl-models", create_if_missing=True)


@app.function(
    image=modal.Image.debian_slim(python_version="3.11").pip_install(
        "scikit-learn==1.5.2", "pandas==2.3.3"
    ),
    volumes={"/models": volume},
    timeout=60,
)
def upload_to_volume(model_b: bytes, features_b: bytes):
    import os

    os.makedirs("/models", exist_ok=True)

    print(f"Writing model ({len(model_b)} bytes) to volume...")
    with open("/models/random_forest_model.pkl", "wb") as f:
        f.write(model_b)

    print(f"Writing features ({len(features_b)} bytes) to volume...")
    with open("/models/features.pkl", "wb") as f:
        f.write(features_b)

    print("Committing volume...")
    volume.commit()

    # Verify
    print("Verifying...")
    with open("/models/random_forest_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("/models/features.pkl", "rb") as f:
        feats = pickle.load(f)

    print(f"Model type: {type(model).__name__}")
    print(f"Features: {feats}")

    return {
        "status": "success",
        "model_type": str(type(model).__name__),
        "features": feats,
    }


print("\nUploading to Modal...")
result = upload_to_volume.remote(model_bytes, features_bytes)

print("\n✓ Upload successful!")
print(f"Model type: {result['model_type']}")
print(f"Features: {len(result['features'])} features")
