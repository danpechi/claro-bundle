"""
Register the Claro Emotion Model in Unity Catalog.
Run via the claro_setup job (Task 1 — register_model).
"""
import base64
import inspect
import io
import os
import struct
import wave

import mlflow
from mlflow.models import ModelSignature
from mlflow.types.schema import ColSpec, Schema

# ── Config ────────────────────────────────────────────────────────────────────

CATALOG    = "main"
SCHEMA     = "claro"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.emotion_speech_model"
# Build experiment path under current user's home — works in any workspace
from databricks.sdk import WorkspaceClient as _WC
_me        = _WC().current_user.me()
EXPERIMENT = f"/Users/{_me.user_name}/claro-emotion-model-registration"

# Resolve emotion_model.py relative to this script.
# Use inspect.getfile() instead of __file__ — Databricks runs scripts via
# exec() which doesn't set __file__, but the compiled code object still
# carries the correct filename.
_HERE            = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
EMOTION_MODEL_PY = os.path.join(_HERE, "emotion_model.py")

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT)

# ── Signature ─────────────────────────────────────────────────────────────────

signature = ModelSignature(
    inputs=Schema([ColSpec("string", "audio_base64")]),
    outputs=Schema([
        ColSpec("string", "emotion"),
        ColSpec("string", "scores_json"),
    ]),
)

# ── Input example (silent 0.5 s WAV, base64) ─────────────────────────────────

def _silent_wav_b64(duration_s: float = 0.5, sr: int = 16000) -> str:
    n   = int(duration_s * sr)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{n}h", *([0] * n)))
    return base64.b64encode(buf.getvalue()).decode()

input_example = {"audio_base64": _silent_wav_b64()}

# ── Log & register ────────────────────────────────────────────────────────────

pip_requirements = [
    "mlflow>=2.14.0",
    "transformers>=4.40.0,<4.51.0",  # 4.51+ has isatty() bug in StreamToLogger env
    "av>=12.0.0",                    # PyAV for M4A/AAC support (ships own FFmpeg)
    "torch>=2.0.0",
    "torchaudio>=2.0.0",
    "librosa>=0.10.0",
    "soundfile>=0.12.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
]

print(f"Logging model → {MODEL_NAME}")
print(f"Model file:     {EMOTION_MODEL_PY}")

with mlflow.start_run(run_name="emotion_model_registration"):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="emotion_model",
        python_model=EMOTION_MODEL_PY,
        pip_requirements=pip_requirements,
        signature=signature,
        input_example=input_example,
        registered_model_name=MODEL_NAME,
    )

print(f"✅  Registered: {MODEL_NAME}")
print(f"    URI:        {model_info.model_uri}")
print(f"    Run ID:     {model_info.run_id}")
print(f"    Version:    {model_info.registered_model_version}")
