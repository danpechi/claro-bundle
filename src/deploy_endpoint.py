"""
Create or update the claro-emotion endpoint to serve the latest registered
version of {catalog}.{schema}.emotion_speech_model_{user_short}.
Run via the claro_setup job (Task 2 — deploy_endpoint).

sys.argv[1] = catalog  (default: main)
sys.argv[2] = schema   (default: claro)
"""
import sys

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

CATALOG = sys.argv[1] if len(sys.argv) > 1 else "main"
SCHEMA  = sys.argv[2] if len(sys.argv) > 2 else "claro"

# Derive the same user suffix used in log_model.py
_w     = WorkspaceClient()
_me    = _w.current_user.me()
_short = "".join(c for c in (_me.user_name or "").split("@")[0] if c.isalnum())[:8]

MODEL_NAME    = f"{CATALOG}.{SCHEMA}.emotion_speech_model_{_short}"
ENDPOINT_NAME = f"claro-emotion-{_short}"

print(f"Model:    {MODEL_NAME}")
print(f"Endpoint: {ENDPOINT_NAME}")

mlflow.set_registry_uri("databricks-uc")

# ── Find latest registered version ───────────────────────────────────────────

client   = mlflow.MlflowClient()
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
if not versions:
    raise RuntimeError(f"No registered versions found for {MODEL_NAME}")

latest_version = str(max(int(v.version) for v in versions))
print(f"Latest model version: {latest_version}")

# ── Create or update endpoint (non-blocking) ──────────────────────────────────

served = [
    ServedEntityInput(
        entity_name=MODEL_NAME,
        entity_version=latest_version,
        scale_to_zero_enabled=True,
        workload_size="Small",
    )
]

try:
    _w.serving_endpoints.get(ENDPOINT_NAME)
    print(f"Endpoint '{ENDPOINT_NAME}' exists — updating to version {latest_version}...")
    _w.serving_endpoints.update_config(name=ENDPOINT_NAME, served_entities=served)
except Exception:
    print(f"Creating endpoint '{ENDPOINT_NAME}' with version {latest_version}...")
    _w.serving_endpoints.create(
        name=ENDPOINT_NAME,
        config=EndpointCoreConfigInput(served_entities=served),
    )

print(f"✅  Endpoint '{ENDPOINT_NAME}' deployment initiated.")
print(f"    It will be READY in ~10-15 minutes.")
print(f"    Check status: Databricks UI → Serving → {ENDPOINT_NAME}")
print(f"    Use this name in the Claro app sidebar: {ENDPOINT_NAME}")
