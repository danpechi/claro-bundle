"""
Create or update the claro-emotion-endpoint to serve the latest registered
version of main.claro.emotion_speech_model.
Run via the claro_setup job (Task 2 — deploy_endpoint).
"""
import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

MODEL_NAME    = "main.claro.emotion_speech_model"
ENDPOINT_NAME = "claro-emotion-endpoint"

mlflow.set_registry_uri("databricks-uc")

# ── Find latest registered version ───────────────────────────────────────────

client   = mlflow.MlflowClient()
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
if not versions:
    raise RuntimeError(f"No registered versions found for {MODEL_NAME}")

latest_version = str(max(int(v.version) for v in versions))
print(f"Latest model version: {latest_version}")

# ── Create or update endpoint (non-blocking) ──────────────────────────────────

w      = WorkspaceClient()
served = [
    ServedEntityInput(
        entity_name=MODEL_NAME,
        entity_version=latest_version,
        scale_to_zero_enabled=True,
        workload_size="Small",
    )
]

try:
    w.serving_endpoints.get(ENDPOINT_NAME)
    print(f"Endpoint '{ENDPOINT_NAME}' exists — updating to version {latest_version}...")
    w.serving_endpoints.update_config(name=ENDPOINT_NAME, served_entities=served)
except Exception:
    print(f"Creating endpoint '{ENDPOINT_NAME}' with version {latest_version}...")
    w.serving_endpoints.create(
        name=ENDPOINT_NAME,
        config=EndpointCoreConfigInput(served_entities=served),
    )

print(f"✅  Endpoint '{ENDPOINT_NAME}' deployment initiated.")
print(f"    It will be READY in ~10-15 minutes.")
print(f"    Check status: Databricks UI → Serving → {ENDPOINT_NAME}")
