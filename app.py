"""
Claro Perú — Customer Service Quality Monitor
==============================================
Streamlit app with three tabs:
  Tab 1 — Quality Monitor: dashboard from {UC_CATALOG}.{UC_SCHEMA}.conversations + .scores
  Tab 2 — Audio Transcription: upload WAV/MP3 → Whisper endpoint (configurable)
  Tab 3 — Emotion Analysis: upload WAV/MP3 → wav2vec2 emotion model endpoint (configurable)
"""

import base64
import configparser
import io
import json
import os
from pathlib import Path

import requests
import streamlit as st

# ── Branding ──────────────────────────────────────────────────────────────

BRAND_NAME   = "Claro Perú"
BRAND_EMOJI  = "📞"
BRAND_COLOR  = "#DA0000"    # Claro red
BRAND_DARK   = "#A80000"    # darker red
BRAND_GRAY   = "#F5F5F5"    # light bg
BRAND_TEXT   = "#1A1A1A"

# ── UC config (from app env vars set by DAB, sidebar can override) ─────────

UC_CATALOG = os.environ.get("UC_CATALOG", "main")
UC_SCHEMA  = os.environ.get("UC_SCHEMA",  "claro")

# Endpoint defaults from env vars injected by DAB at deploy time
DEFAULT_EMOTION_ENDPOINT = os.environ.get("EMOTION_ENDPOINT_NAME", "")
DEFAULT_WHISPER_ENDPOINT = os.environ.get("WHISPER_ENDPOINT_NAME", "")

# ── Table suffix (injected by DAB via UC_USER_SUFFIX env var) ─────────────
# Empty in customer/single-user workspaces; set to e.g. "_dan" by SEs
# deploying into a shared workspace via: databricks bundle deploy --var user_suffix=_dan

UC_USER_SUFFIX = os.environ.get("UC_USER_SUFFIX", "")

# ── Credentials ───────────────────────────────────────────────────────────
# In Databricks Apps the SDK picks up DATABRICKS_HOST + DATABRICKS_CLIENT_ID +
# DATABRICKS_CLIENT_SECRET and does OAuth M2M automatically.
# For local dev it falls back to ~/.databrickscfg or DATABRICKS_TOKEN.

@st.cache_resource
def _get_databricks_client():
    from databricks.sdk import WorkspaceClient
    return WorkspaceClient()


def _auth():
    """Return (host, headers) with a fresh OAuth token on every call."""
    try:
        w = _get_databricks_client()
        host = w.config.host or ""
        if not host.startswith("http"):
            host = "https://" + host
        headers = {**w.config.authenticate(), "Content-Type": "application/json"}
        return host.rstrip("/"), headers
    except Exception:
        # Local dev fallback
        cfg = configparser.ConfigParser()
        cfg.read(Path.home() / ".databrickscfg")
        host  = cfg.get("DEFAULT", "host",  fallback="") or os.environ.get("DATABRICKS_HOST", "")
        token = cfg.get("DEFAULT", "token", fallback="") or os.environ.get("DATABRICKS_TOKEN", "")
        if host and not host.startswith("http"):
            host = "https://" + host
        return host.rstrip("/"), {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# ── SQL helper ─────────────────────────────────────────────────────────────

@st.cache_resource
def _get_wh_id() -> str:
    host, headers = _auth()
    url = f"{host}/api/2.0/sql/warehouses"
    r   = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    whs     = r.json().get("warehouses", [])
    running = [w for w in whs if w.get("state") == "RUNNING"]
    return (running or whs)[0]["id"]


def sql_run(query: str) -> list[dict]:
    host, headers = _auth()
    wh_id = _get_wh_id()
    url   = f"{host}/api/2.0/sql/statements"
    payload = {
        "statement":       query,
        "warehouse_id":    wh_id,
        "wait_timeout":    "30s",
        "on_wait_timeout": "CANCEL",
        "disposition":     "INLINE",
        "format":          "JSON_ARRAY",
    }
    r = requests.post(url, json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    body   = r.json()
    status = body.get("status", {}).get("state", "")
    if status != "SUCCEEDED":
        raise RuntimeError(body.get("status", {}).get("error", {}).get("message", body))
    cols = [c["name"] for c in body.get("manifest", {}).get("schema", {}).get("columns", [])]
    rows = body.get("result", {}).get("data_array", []) or []
    return [dict(zip(cols, row)) for row in rows]


# ── Audio helpers ─────────────────────────────────────────────────────────

def to_wav_bytes(raw: bytes) -> bytes:
    """Convert any audio format (M4A, MP3, OGG, …) to 16kHz mono WAV bytes using PyAV."""
    import av
    import struct
    import wave as wave_mod

    in_buf  = io.BytesIO(raw)
    out_buf = io.BytesIO()

    with av.open(in_buf) as container:
        stream = next(s for s in container.streams if s.type == "audio")
        sr     = stream.sample_rate or 16000

        graph = av.filter.Graph()
        src   = graph.add_abuffer(
            sample_rate=sr,
            format="fltp",
            channels=stream.channels,
            layout=stream.layout.name,
        )
        aresample = graph.add("aresample", "16000")
        aformat   = graph.add("aformat", "sample_rates=16000:channel_layouts=mono:sample_fmts=s16")
        sink      = graph.add("abuffersink")
        src.link_to(aresample)
        aresample.link_to(aformat)
        aformat.link_to(sink)
        graph.configure()

        samples = []
        for packet in container.demux(stream):
            for frame in packet.decode():
                frame.pts = None
                graph.push(frame)
                try:
                    while True:
                        f = graph.pull()
                        samples.append(bytes(f.planes[0]))
                except av.AVError:
                    pass
        graph.push(None)
        try:
            while True:
                f = graph.pull()
                samples.append(bytes(f.planes[0]))
        except av.AVError:
            pass

    pcm = b"".join(samples)
    with wave_mod.open(out_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm)

    return out_buf.getvalue()


def encode_audio(uploaded_file) -> str:
    """Read uploaded file, convert to WAV if needed, return base64 string."""
    uploaded_file.seek(0)
    raw  = uploaded_file.read()
    name = getattr(uploaded_file, "name", "").lower()

    # Convert non-WAV formats to 16kHz mono WAV so the model always gets clean input
    if not name.endswith(".wav"):
        try:
            raw = to_wav_bytes(raw)
        except Exception:
            pass  # fall through — let the model try its own decoders

    return base64.b64encode(raw).decode()


def call_emotion_endpoint(audio_b64: str, endpoint_name: str) -> dict:
    """Call the emotion endpoint with base64 audio, return {emotion, scores}."""
    host, headers = _auth()
    url     = f"{host}/serving-endpoints/{endpoint_name}/invocations"
    payload = {"dataframe_records": [{"audio_base64": audio_b64}]}
    r = requests.post(url, json=payload, headers=headers, timeout=120)
    r.raise_for_status()
    body = r.json()
    # MLflow pyfunc returns {"predictions": [...]}
    predictions = body.get("predictions", body.get("outputs", [{}]))
    if isinstance(predictions, list) and predictions:
        return predictions[0]
    return predictions


def call_whisper_endpoint(audio_b64: str, endpoint_name: str) -> str:
    """Call the Whisper MLflow pyfunc endpoint; returns transcription text.

    Uses dataframe_split format matching the whisper-dan deployment.
    """
    host, headers = _auth()
    url = f"{host}/serving-endpoints/{endpoint_name}/invocations"
    payload = {
        "dataframe_split": {
            "columns": [0],
            "data":    [[audio_b64]],
        }
    }
    r = requests.post(url, json=payload, headers=headers, timeout=120)
    r.raise_for_status()
    body  = r.json()
    preds = body.get("predictions", [])
    if preds:
        p = preds[0]
        if isinstance(p, dict):
            return p.get("text", str(p)).strip()
        return str(p).strip()
    # Fallback: OpenAI-compatible choices
    choices = body.get("choices", [])
    if choices:
        return choices[0].get("message", {}).get("content", "").strip()
    return str(body)


# ── Data loaders ──────────────────────────────────────────────────────────

@st.cache_data(ttl=120, show_spinner=False)
def load_scores(catalog: str, schema: str, suffix: str) -> list[dict]:
    table = f"{catalog}.{schema}.scores_{suffix}" if suffix else f"{catalog}.{schema}.scores"
    try:
        return sql_run(f"SELECT * FROM {table} ORDER BY created_at DESC")
    except Exception:
        return []


@st.cache_data(ttl=120, show_spinner=False)
def load_transcript(conv_id: str, catalog: str, schema: str, suffix: str) -> list[dict]:
    table = f"{catalog}.{schema}.conversations_{suffix}" if suffix else f"{catalog}.{schema}.conversations"
    try:
        return sql_run(
            f"SELECT turn_num, role, content FROM {table} "
            f"WHERE conv_id = '{conv_id}' ORDER BY turn_num"
        )
    except Exception:
        return []


# ── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=f"{BRAND_NAME} — Quality Monitor",
    page_icon=BRAND_EMOJI,
    layout="wide",
)

st.markdown(f"""
<style>
  [data-testid="stAppViewContainer"] > .main {{
      border-top: 5px solid {BRAND_COLOR};
  }}
  .claro-header {{
      color: {BRAND_COLOR};
      font-size: 1.8rem;
      font-weight: 800;
      letter-spacing: -0.5px;
  }}
  .claro-sub {{
      color: #555;
      font-size: 0.95rem;
      margin-top: -8px;
  }}
  /* KPI cards */
  .kpi-card {{
      background: #fff;
      border: 1px solid #e0e0e0;
      border-top: 4px solid {BRAND_COLOR};
      border-radius: 8px;
      padding: 18px 20px;
      text-align: center;
  }}
  .kpi-value {{
      font-size: 2.2rem;
      font-weight: 800;
      color: {BRAND_COLOR};
      line-height: 1.1;
  }}
  .kpi-label {{
      font-size: 0.78rem;
      color: #777;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-top: 4px;
  }}
  /* Compliance badge */
  .badge-yes {{
      background: #e8f5e9;
      color: #2e7d32;
      border-radius: 12px;
      padding: 2px 10px;
      font-size: 0.8rem;
      font-weight: 700;
  }}
  .badge-no {{
      background: #ffebee;
      color: #c62828;
      border-radius: 12px;
      padding: 2px 10px;
      font-size: 0.8rem;
      font-weight: 700;
  }}
  /* Conversation row */
  .conv-card {{
      background: #fff;
      border: 1px solid #ddd;
      border-radius: 8px;
      padding: 12px 16px;
      margin-bottom: 8px;
      cursor: pointer;
  }}
  /* Transcript bubbles */
  .bubble-user {{
      background: {BRAND_GRAY};
      border-radius: 14px 14px 14px 4px;
      padding: 10px 14px;
      margin: 6px 0;
      max-width: 80%;
      font-size: 0.9rem;
  }}
  .bubble-agent {{
      background: {BRAND_COLOR};
      color: #fff;
      border-radius: 14px 14px 4px 14px;
      padding: 10px 14px;
      margin: 6px 0;
      max-width: 80%;
      margin-left: auto;
      font-size: 0.9rem;
  }}
  .role-label {{
      font-size: 0.72rem;
      font-weight: 700;
      color: #888;
      margin-bottom: 2px;
  }}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"## {BRAND_EMOJI} {BRAND_NAME}")
    st.divider()

    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.markdown("### 🗄️ Data Source")
    uc_catalog = st.text_input("UC Catalog", value=UC_CATALOG)
    uc_schema  = st.text_input("UC Schema",  value=UC_SCHEMA)
    uc_suffix  = st.text_input(
        "Table suffix",
        value=UC_USER_SUFFIX,
        help="Suffix appended to table names (e.g. _dan). Leave empty for default deployment.",
    )

    st.divider()
    st.markdown("### 📊 Monitor Filters")

    channel_opts = ["All", "Call Center", "WhatsApp", "Website", "In-Branch"]
    sel_channel  = st.selectbox("Channel", channel_opts)

    tier_opts = ["All", "compliant", "partial", "non_compliant"]
    sel_tier  = st.selectbox("Agent quality tier", tier_opts)

    compliance_opts = ["All", "yes", "no"]
    sel_compliance  = st.selectbox("Protocol compliance", compliance_opts)

    reason_opts = ["All", "billing", "tech_support", "plan_upgrade", "complaint", "general_inquiry"]
    sel_reason  = st.selectbox("Contact reason", reason_opts)

    st.divider()
    st.markdown("### 🎤 Whisper Settings")
    whisper_endpoint = st.text_input(
        "Whisper endpoint name",
        value=DEFAULT_WHISPER_ENDPOINT,
        help="Name of the Databricks serving endpoint running Whisper",
    )

    st.divider()
    st.markdown("### 😊 Emotion Settings")
    emotion_endpoint = st.text_input(
        "Emotion endpoint name",
        value=DEFAULT_EMOTION_ENDPOINT,
        help="Name of the Databricks serving endpoint running the wav2vec2 emotion model",
    )

    st.divider()
    st.caption("Data sourced from synthetic Claro Perú conversations stored in Delta.")

# ── Page header ────────────────────────────────────────────────────────────

st.markdown(f'<p class="claro-header">{BRAND_EMOJI} {BRAND_NAME}</p>', unsafe_allow_html=True)
st.markdown('<p class="claro-sub">Plataforma de Calidad · Transcripción · Análisis de Emoción</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Monitor de Calidad", "🎤 Transcripción de Audio", "😊 Análisis de Emoción"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Quality Monitor
# ═══════════════════════════════════════════════════════════════════════════

with tab1:

    # ── Load data ────────────────────────────────────────────────────────

    all_scores = load_scores(uc_catalog, uc_schema, uc_suffix)

    if not all_scores:
        _tbl = f"{uc_catalog}.{uc_schema}.scores{uc_suffix}"
        st.warning(
            f"No data found in `{_tbl}`. "
            "Run the **claro_setup** job first to populate the tables."
        )
    else:

        # ── Apply filters ────────────────────────────────────────────────────

        filtered = all_scores
        if sel_channel    != "All": filtered = [r for r in filtered if r["channel"]             == sel_channel]
        if sel_tier       != "All": filtered = [r for r in filtered if r["quality_tier"]        == sel_tier]
        if sel_compliance != "All": filtered = [r for r in filtered if r["protocol_compliance"] == sel_compliance]
        if sel_reason     != "All": filtered = [r for r in filtered if r["contact_reason"]      == sel_reason]

        st.markdown(
            f'<p class="claro-sub">Análisis de calidad de servicio al cliente · '
            f'{len(filtered)} conversaciones (de {len(all_scores)} total)</p>',
            unsafe_allow_html=True,
        )
        st.divider()

        from collections import Counter

        # ── KPIs ─────────────────────────────────────────────────────────────

        REASON_LABELS = {
            "billing":        "💳 Facturación",
            "tech_support":   "🔧 Soporte Técnico",
            "plan_upgrade":   "📶 Plan / Upgrade",
            "complaint":      "⚠️ Reclamo",
            "general_inquiry":"ℹ️ Consulta General",
        }

        total      = len(filtered)
        comp_yes   = sum(1 for r in filtered if r.get("protocol_compliance") == "yes")
        comp_rate  = f"{100*comp_yes//total}%" if total else "—"
        avg_qual   = (sum(float(r.get("service_quality_score") or 0) for r in filtered) / total) if total else 0

        top_reason_counter = Counter(r.get("contact_reason", "?") for r in filtered)
        top_reason = top_reason_counter.most_common(1)[0][0].replace("_", " ").title() if total else "—"

        col1, col2, col3, col4 = st.columns(4)
        kpis = [
            (col1, str(total),                    "Conversaciones"),
            (col2, comp_rate,                     "Cumplimiento de Protocolo"),
            (col3, f"{avg_qual:.1f}/5",           "Calidad de Servicio"),
            (col4, top_reason.replace("_", " "), "Motivo Principal de Contacto"),
        ]
        for col, val, label in kpis:
            with col:
                st.markdown(
                    f'<div class="kpi-card">'
                    f'<div class="kpi-value">{val}</div>'
                    f'<div class="kpi-label">{label}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Charts ────────────────────────────────────────────────────────────

        chart_col1, chart_col2, chart_col3 = st.columns(3)
        channels = ["Call Center", "WhatsApp", "Website", "In-Branch"]

        with chart_col1:
            st.markdown("#### Cumplimiento por Canal")
            ch_data = {}
            for ch in channels:
                rows    = [r for r in filtered if r.get("channel") == ch]
                yes_pct = (sum(1 for r in rows if r.get("protocol_compliance") == "yes") / len(rows) * 100) if rows else 0
                ch_data[ch] = {"yes_pct": yes_pct, "n": len(rows)}
            for ch, d in ch_data.items():
                if d["n"] == 0:
                    continue
                st.markdown(f"**{ch}** ({d['n']} conv.)")
                st.progress(int(d["yes_pct"]) / 100, text=f"{d['yes_pct']:.0f}% cumple")

        with chart_col2:
            st.markdown("#### Calidad por Canal")
            for ch in channels:
                rows = [r for r in filtered if r.get("channel") == ch]
                if not rows:
                    continue
                avg = sum(float(r.get("service_quality_score") or 0) for r in rows) / len(rows)
                st.markdown(f"**{ch}** → `{avg:.1f}/5` {'⭐' * round(avg)}")
                st.progress(avg / 5)

        with chart_col3:
            st.markdown("#### Motivos de Contacto")
            reason_counts = Counter(r.get("contact_reason", "?") for r in filtered)
            for reason, count in reason_counts.most_common():
                label = REASON_LABELS.get(reason, reason)
                pct   = count / total * 100 if total else 0
                st.markdown(f"{label}")
                st.progress(pct / 100, text=f"{count} ({pct:.0f}%)")

        st.divider()

        # ── Conversation explorer ──────────────────────────────────────────────

        st.markdown("### 🔍 Explorador de Conversaciones")
        st.caption("Selecciona una conversación para ver la transcripción completa y los detalles de evaluación.")

        TIER_COLORS = {
            "compliant":     "🟢",
            "partial":       "🟡",
            "non_compliant": "🔴",
        }

        if not filtered:
            st.info("No hay conversaciones con los filtros seleccionados.")
        else:
            header_cols = st.columns([1.2, 1.5, 1, 1.2, 1, 1.2])
            for col, label in zip(header_cols, ["Conv ID", "Canal", "Escenario", "Cumplimiento", "Calidad", "Motivo"]):
                col.markdown(f"**{label}**")
            st.markdown("---")

            for row in filtered:
                conv_id    = row["conv_id"]
                channel    = row.get("channel", "")
                scenario   = row.get("scenario", "").replace("_", " ").title()
                tier       = row.get("quality_tier", "")
                compliance = row.get("protocol_compliance", "?")
                qual       = float(row.get("service_quality_score") or 0)
                reason     = REASON_LABELS.get(row.get("contact_reason", ""), row.get("contact_reason", ""))

                comp_badge = (
                    f'<span class="badge-yes">✓ Cumple</span>'
                    if compliance == "yes"
                    else f'<span class="badge-no">✗ Incumple</span>'
                )
                tier_dot = TIER_COLORS.get(tier, "⚪")

                row_cols = st.columns([1.2, 1.5, 1, 1.2, 1, 1.2])
                row_cols[0].markdown(f"`{conv_id}`")
                row_cols[1].markdown(channel)
                row_cols[2].markdown(scenario)
                row_cols[3].markdown(comp_badge, unsafe_allow_html=True)
                row_cols[4].markdown(f"{'⭐' * round(qual)} `{qual:.1f}`")
                row_cols[5].markdown(reason)

                with st.expander(f"Ver transcripción — {tier_dot} {tier.replace('_',' ').title()}"):
                    eval_col1, eval_col2, eval_col3 = st.columns(3)
                    with eval_col1:
                        st.markdown("**Cumplimiento de Protocolo**")
                        st.markdown(comp_badge, unsafe_allow_html=True)
                        if row.get("compliance_rationale"):
                            st.caption(row["compliance_rationale"][:200])
                    with eval_col2:
                        st.markdown("**Calidad de Servicio**")
                        st.markdown(f"{'⭐' * round(qual)} **{qual:.1f}/5**")
                        if row.get("quality_rationale"):
                            st.caption(row["quality_rationale"][:200])
                    with eval_col3:
                        st.markdown("**Motivo de Contacto**")
                        st.markdown(f"**{reason}**")
                        if row.get("reason_rationale"):
                            st.caption(row["reason_rationale"][:200])

                    st.markdown("---")
                    st.markdown("**Transcripción**")
                    turns = load_transcript(conv_id, uc_catalog, uc_schema, uc_suffix)
                    if turns:
                        for turn in turns:
                            role    = turn["role"]
                            content = turn["content"]
                            if role == "user":
                                st.markdown(
                                    f'<div class="role-label">CLIENTE</div>'
                                    f'<div class="bubble-user">{content}</div>',
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    f'<div style="text-align:right">'
                                    f'<div class="role-label">AGENTE CLARO</div>'
                                    f'<div class="bubble-agent">{content}</div>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                    else:
                        st.info("Transcripción no disponible.")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Audio Transcription (Whisper)
# ═══════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("### 🎤 Transcripción de Audio")

    if not whisper_endpoint:
        st.warning(
            "No Whisper endpoint configured. Enter the endpoint name in the sidebar under **Whisper Settings**.",
            icon="⚠️",
        )
    else:
        st.markdown(
            f"Sube un archivo de audio (WAV o MP3) y transcríbelo usando el endpoint "
            f"**`{whisper_endpoint}`** vía Databricks Model Serving."
        )
        st.info(
            "💡 Puedes cambiar el nombre del endpoint de Whisper en la barra lateral.",
            icon="ℹ️",
        )

    audio_file_whisper = st.file_uploader(
        "Subir archivo de audio",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
        key="whisper_uploader",
    )

    if audio_file_whisper is not None:
        st.audio(audio_file_whisper)
        if st.button("🔤 Transcribir", key="btn_whisper", disabled=not whisper_endpoint):
            with st.spinner("Transcribiendo con Whisper..."):
                try:
                    audio_b64 = encode_audio(audio_file_whisper)
                    text = call_whisper_endpoint(audio_b64, whisper_endpoint)
                    st.success("Transcripción completada")
                    st.markdown("#### Transcripción")
                    st.markdown(
                        f'<div style="background:#f9f9f9;border-left:4px solid {BRAND_COLOR};'
                        f'padding:16px;border-radius:4px;font-size:1rem;line-height:1.6;">'
                        f'{text}</div>',
                        unsafe_allow_html=True,
                    )
                    # Copy-friendly text area
                    st.text_area("Texto plano (para copiar)", value=text, height=120, key="whisper_result")
                except Exception as e:
                    st.error(f"Error al transcribir: {e}")
                    st.caption("Verifica que el endpoint esté disponible (READY) y que el nombre sea correcto.")
    else:
        st.markdown(
            '<div style="border:2px dashed #ccc;border-radius:8px;padding:40px;'
            'text-align:center;color:#999;">Sube un archivo de audio para comenzar</div>',
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Emotion Analysis (wav2vec2)
# ═══════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("### 😊 Análisis de Emoción")

    if not emotion_endpoint:
        st.warning(
            "No emotion endpoint configured. Enter the endpoint name in the sidebar under **Emotion Settings**. "
            "Run the `claro_setup` job first to deploy the endpoint — it will appear in the job logs.",
            icon="⚠️",
        )
    else:
        st.markdown(
            f"Sube un archivo de audio (WAV, 16 kHz recomendado) y detecta la emoción "
            f"usando el modelo **`ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`** "
            f"desplegado en el endpoint **`{emotion_endpoint}`**."
        )

    EMOTION_CONFIG = {
        "angry":     {"emoji": "😠", "color": "#e53935", "label": "Enojo"},
        "calm":      {"emoji": "😌", "color": "#26a69a", "label": "Calma"},
        "disgust":   {"emoji": "🤢", "color": "#8d6e63", "label": "Disgusto"},
        "fearful":   {"emoji": "😨", "color": "#7b1fa2", "label": "Miedo"},
        "happy":     {"emoji": "😊", "color": "#43a047", "label": "Alegría"},
        "neutral":   {"emoji": "😐", "color": "#757575", "label": "Neutral"},
        "sad":       {"emoji": "😢", "color": "#1565c0", "label": "Tristeza"},
        "surprised": {"emoji": "😲", "color": "#f57c00", "label": "Sorpresa"},
    }

    audio_file_emotion = st.file_uploader(
        "Subir archivo de audio",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
        key="emotion_uploader",
    )

    if audio_file_emotion is not None:
        st.audio(audio_file_emotion)
        if st.button("🔍 Analizar Emoción", key="btn_emotion", disabled=not emotion_endpoint):
            with st.spinner("Analizando emoción..."):
                try:
                    audio_b64  = encode_audio(audio_file_emotion)
                    result     = call_emotion_endpoint(audio_b64, emotion_endpoint)
                    emotion    = result.get("emotion", "unknown")
                    scores_raw = result.get("scores_json", "{}")
                    scores     = json.loads(scores_raw) if isinstance(scores_raw, str) else scores_raw

                    cfg = EMOTION_CONFIG.get(emotion, {"emoji": "❓", "color": BRAND_COLOR, "label": emotion.title()})

                    st.success("Análisis completado")

                    # Main result card
                    st.markdown(
                        f'<div style="background:#fff;border:2px solid {cfg["color"]};'
                        f'border-radius:12px;padding:24px;text-align:center;margin:16px 0;">'
                        f'<div style="font-size:3rem;">{cfg["emoji"]}</div>'
                        f'<div style="font-size:1.8rem;font-weight:800;color:{cfg["color"]};">'
                        f'{cfg["label"]}</div>'
                        f'<div style="color:#777;font-size:0.85rem;margin-top:4px;">'
                        f'Emoción detectada: <strong>{emotion}</strong></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # Score bars
                    if scores:
                        st.markdown("#### Probabilidades por emoción")
                        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                        for emo, prob in sorted_scores:
                            emo_cfg = EMOTION_CONFIG.get(emo, {"emoji": "❓", "label": emo.title()})
                            is_top  = emo == emotion
                            label   = f"{'**' if is_top else ''}{emo_cfg['emoji']} {emo_cfg['label']}{'**' if is_top else ''}"
                            st.markdown(label)
                            st.progress(float(prob), text=f"{float(prob)*100:.1f}%")

                except Exception as e:
                    st.error(f"Error al analizar emoción: {e}")
                    st.caption(
                        f"Verifica que el endpoint **{emotion_endpoint}** esté en estado READY. "
                        "El despliegue inicial puede tardar ~15 minutos."
                    )
    else:
        st.markdown(
            '<div style="border:2px dashed #ccc;border-radius:8px;padding:40px;'
            'text-align:center;color:#999;">Sube un archivo de audio para comenzar</div>',
            unsafe_allow_html=True,
        )

    with st.expander("ℹ️ Sobre el modelo de emoción"):
        st.markdown(
            f"""
**Modelo:** `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`

**Emociones detectadas:** angry, calm, disgust, fearful, happy, neutral, sad, surprised

**Entrada esperada:** Audio WAV mono, 16 kHz (el modelo resamplea automáticamente si es necesario)

**Endpoint:** `{emotion_endpoint or "<configurar en sidebar>"}` (Databricks Model Serving)

**Formato de entrada al endpoint:**
```json
{{"dataframe_records": [{{"audio_base64": "<base64-encoded WAV>"}}]}}
```

**Formato de respuesta:**
```json
{{"predictions": [{{"emotion": "happy", "scores_json": "{{\\"happy\\": 0.82, \\"sad\\": 0.05, ...}}"}}]}}
```
"""
        )
