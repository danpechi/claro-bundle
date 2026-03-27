"""
Claro Emotion Model — MLflow PyFunc wrapper for
ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition

Input:  {"audio_base64": "<base64-encoded WAV/MP3 bytes>"}
Output: {"emotion": "happy", "scores": {"happy": 0.82, "sad": 0.05, ...}}
"""

import base64
import io
import json

import mlflow
import numpy as np
import pandas as pd


class EmotionModel(mlflow.pyfunc.PythonModel):

    MODEL_ID = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    TARGET_SR = 16_000  # wav2vec2 expects 16 kHz

    def load_context(self, context):
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        import torch

        self.extractor = AutoFeatureExtractor.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForAudioClassification.from_pretrained(self.MODEL_ID)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.labels = self.model.config.id2label  # {0: "angry", 1: "happy", ...}

    def _decode_audio(self, audio_b64: str) -> np.ndarray:
        """Decode base64 audio → 16 kHz float32 numpy array.

        Tries (in order): soundfile → librosa → PyAV (handles M4A/AAC).
        """
        import librosa
        import soundfile as sf

        raw = base64.b64decode(audio_b64)
        buf = io.BytesIO(raw)

        audio, sr = None, None

        # 1. soundfile (WAV, FLAC, OGG, AIFF, …)
        try:
            audio, sr = sf.read(buf)
        except Exception:
            pass

        # 2. librosa (MP3, WebM, …)
        if audio is None:
            try:
                buf.seek(0)
                audio, sr = librosa.load(buf, sr=None, mono=True)
            except Exception:
                pass

        # 3. PyAV (M4A / AAC / MP4 audio — ships its own FFmpeg binaries)
        if audio is None:
            try:
                import av
                buf.seek(0)
                with av.open(buf) as container:
                    stream = next(s for s in container.streams if s.type == "audio")
                    sr = stream.sample_rate
                    frames = []
                    for frame in container.decode(stream):
                        frames.append(frame.to_ndarray())
                audio = np.concatenate(frames, axis=-1).astype(np.float32)
                # av returns (channels, samples) — flatten to mono
                if audio.ndim > 1:
                    audio = audio.mean(axis=0)
                # Normalise int16 range if needed
                if audio.max() > 1.0:
                    audio = audio / 32768.0
            except Exception as e:
                raise ValueError(f"Could not decode audio (tried soundfile, librosa, PyAV): {e}")

        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample to 16 kHz if needed
        if sr != self.TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.TARGET_SR)

        return audio.astype(np.float32)

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        import torch

        results = []
        for _, row in model_input.iterrows():
            audio_b64 = row.get("audio_base64", "")
            if not audio_b64:
                results.append({"emotion": "unknown", "scores_json": "{}"})
                continue

            audio = self._decode_audio(audio_b64)

            inputs = self.extractor(
                audio,
                sampling_rate=self.TARGET_SR,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits

            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
            scores = {self.labels[i]: float(p) for i, p in enumerate(probs)}
            top_emotion = max(scores, key=scores.get)

            results.append({
                "emotion":     top_emotion,
                "scores_json": json.dumps({k: round(v, 4) for k, v in scores.items()}),
            })

        return pd.DataFrame(results)


# Required for file-based logging
mlflow.models.set_model(EmotionModel())
