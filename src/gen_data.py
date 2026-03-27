"""
Generate synthetic Claro Perú conversations and quality scores.
Writes Delta tables {catalog}.{schema}.conversations_{short} and
{catalog}.{schema}.scores_{short} — user-specific to avoid collisions.

Run via the claro_setup job (parallel with register_model).

sys.argv[1] = catalog  (default: main)
sys.argv[2] = schema   (default: claro)
"""
import inspect
import sys
from datetime import datetime, timedelta

from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession

CATALOG = sys.argv[1] if len(sys.argv) > 1 else "main"
SCHEMA  = sys.argv[2] if len(sys.argv) > 2 else "claro"

_w  = WorkspaceClient()
_me = _w.current_user.me()
_short = "".join(c for c in (_me.user_name or "").split("@")[0] if c.isalnum())[:8]

CONV_TABLE   = f"{CATALOG}.{SCHEMA}.conversations_{_short}"
SCORES_TABLE = f"{CATALOG}.{SCHEMA}.scores_{_short}"

# Ensure schema exists
try:
    _w.schemas.get(f"{CATALOG}.{SCHEMA}")
except Exception:
    _w.schemas.create(name=SCHEMA, catalog_name=CATALOG)
    print(f"✅ Schema {CATALOG}.{SCHEMA} created.")

print(f"Writing conversations → {CONV_TABLE}")
print(f"Writing scores        → {SCORES_TABLE}")

spark = SparkSession.builder.getOrCreate()

# ── Synthetic conversations ────────────────────────────────────────────────
# 15 pre-baked dialogues across 5 scenarios × 4 channels × 3 quality tiers.
# Structure: (conv_id, channel, scenario, quality_tier, [(role, content), ...])

_BASE_TS = datetime(2026, 1, 15, 9, 0, 0)

CONVERSATIONS = [
    ("CONV-001", "Call Center", "billing", "compliant", [
        ("user",      "Buenos días, tengo un cargo de S/89 en mi factura que no reconozco."),
        ("assistant", "Buenos días, le atiende María García de Claro Perú. ¿Me puede proporcionar su DNI para verificar su cuenta?"),
        ("user",      "Sí, mi DNI es 45678901."),
        ("assistant", "Gracias. Verifico que ese cargo corresponde a un servicio adicional activado el 3 de enero. Entiendo la confusión. ¿Desea que lo cancele y gestione el reembolso?"),
        ("user",      "Sí, no lo activé yo."),
        ("assistant", "Listo, he anulado el cargo y solicitado el reembolso en 5 días hábiles. ¿Hay algo más en que pueda ayudarle? Que tenga un excelente día."),
    ]),
    ("CONV-002", "Call Center", "tech_support", "non_compliant", [
        ("user",      "Mi internet lleva 3 días caído, necesito solución urgente."),
        ("assistant", "Mande un técnico, tiene que esperar."),
        ("user",      "¿Cuánto tiempo? Trabajo desde casa y necesito la conexión."),
        ("assistant", "No sé, depende de la zona. Llame mañana."),
    ]),
    ("CONV-003", "Call Center", "plan_upgrade", "compliant", [
        ("user",      "Quiero contratar fibra óptica con TV para mi hogar."),
        ("assistant", "Con mucho gusto, le atiende Carlos López de Claro Perú. ¿Me facilita su DNI para verificar disponibilidad en su zona?"),
        ("user",      "12345678."),
        ("assistant", "Perfecto. En su dirección hay cobertura de 200 Mbps con 150 canales de TV por S/129 al mes. ¿Le parece bien? Puedo agendar la instalación esta semana."),
        ("user",      "Sí, excelente."),
        ("assistant", "Anotado. Le llegará un SMS con la confirmación y horario del técnico. Gracias por elegir Claro, que tenga buen día."),
    ]),
    ("CONV-004", "Call Center", "complaint", "partial", [
        ("user",      "Me prometieron un 20% de descuento que no aparece en mi factura."),
        ("assistant", "Voy a revisar su cuenta. ¿Número de línea?"),
        ("user",      "987654321."),
        ("assistant", "Veo el plan pero no hay registro de esa promoción. Voy a escalar el caso al área de retención."),
        ("user",      "¿Cuánto demora?"),
        ("assistant", "De 3 a 5 días hábiles."),
    ]),
    ("CONV-005", "WhatsApp", "plan_upgrade", "compliant", [
        ("user",      "Hola, quiero cambiar mi plan de 15 GB a uno con más datos."),
        ("assistant", "¡Hola! Buenos días, soy Ana de Claro Perú. Con gusto le ayudo. ¿Me comparte su DNI para revisar su cuenta?"),
        ("user",      "45123678"),
        ("assistant", "Gracias. Tiene disponible un plan de 60 GB con llamadas ilimitadas por S/20 adicionales al mes. ¿Desea activarlo ahora mismo?"),
        ("user",      "Sí perfecto."),
        ("assistant", "¡Listo! Su plan ya fue actualizado. Recibirá un SMS de confirmación en minutos. ¿Hay algo más que necesite? 😊"),
    ]),
    ("CONV-006", "WhatsApp", "billing", "partial", [
        ("user",      "Recargué S/20 hace una hora y no veo el saldo reflejado."),
        ("assistant", "¿Su número de línea?"),
        ("user",      "987123456"),
        ("assistant", "El saldo aparece pendiente de acreditación, puede demorar hasta 2 horas."),
        ("user",      "¿Y si pasan las 2 horas y no aparece?"),
        ("assistant", "Nos escribe de nuevo."),
    ]),
    ("CONV-007", "WhatsApp", "complaint", "non_compliant", [
        ("user",      "Mi velocidad es 8 Mbps y contraté 100 Mbps. Tengo capturas de Speedtest."),
        ("assistant", "Las velocidades pueden variar."),
        ("user",      "No es una variación, es diez veces menos. Quiero una solución o compensación."),
        ("assistant", "Le mando un ticket."),
    ]),
    ("CONV-008", "Website", "general_inquiry", "compliant", [
        ("user",      "¿Tienen cobertura 5G en San Borja? Estoy evaluando cambiarme de Movistar."),
        ("assistant", "¡Bienvenido a Claro Perú! Soy Pedro, con gusto le informo. ¿Podría indicarme su dirección exacta para verificar disponibilidad?"),
        ("user",      "Av. Aviación 2847, San Borja."),
        ("assistant", "Excelente. Su dirección cuenta con cobertura 5G activa. Tenemos planes desde S/89/mes con datos ilimitados 5G. ¿Le gustaría información sobre portabilidad?"),
        ("user",      "Sí, me interesa mucho."),
        ("assistant", "Le comparto el enlace del proceso; puede completarlo en 24 horas conservando su número. Estamos para servirle cuando decida. ¡Que tenga un excelente día!"),
    ]),
    ("CONV-009", "Website", "plan_upgrade", "partial", [
        ("user",      "Quiero portarme de Entel a Claro hoy."),
        ("assistant", "Para portabilidad necesita su DNI y número Entel."),
        ("user",      "DNI 78901234, número 987654321."),
        ("assistant", "Puede iniciar el trámite en claro.com.pe/portabilidad. Demora 24 horas."),
        ("user",      "¿Pierdo mi número?"),
        ("assistant", "No, conserva el mismo número."),
    ]),
    ("CONV-010", "Website", "billing", "non_compliant", [
        ("user",      "Suspendieron mi línea, necesito reactivarla urgente."),
        ("assistant", "Pague su deuda primero."),
        ("user",      "¿Cuánto debo y cómo pago?"),
        ("assistant", "Revise su app de Claro."),
    ]),
    ("CONV-011", "In-Branch", "tech_support", "compliant", [
        ("user",      "Necesito reponer mi SIM dañada sin perder mi número."),
        ("assistant", "Con gusto le ayudo. Soy Luis de Claro Perú. ¿Me facilita su DNI y el número a recuperar?"),
        ("user",      "DNI 45678912, número 987000111."),
        ("assistant", "Perfecto, verifico su identidad. Voy a procesar la reposición ahora mismo; no perderá contactos ni número. Tardará 5 minutos."),
        ("user",      "Gracias."),
        ("assistant", "Aquí tiene su nueva SIM activada. Puede tardar 10 minutos en tener señal. ¿Necesita ayuda con algo más? Fue un placer atenderle."),
    ]),
    ("CONV-012", "In-Branch", "plan_upgrade", "partial", [
        ("user",      "Quiero contratar un plan empresarial para 12 líneas con centralita."),
        ("assistant", "Para empresas necesita hablar con nuestro asesor de negocios."),
        ("user",      "¿Está aquí en la tienda hoy?"),
        ("assistant", "Hoy no está disponible, puede venir mañana o llamar al 123."),
        ("user",      "¿Me puede dar algo de información mientras tanto?"),
        ("assistant", "Le dejo este folleto con los planes empresariales."),
    ]),
    ("CONV-013", "In-Branch", "complaint", "compliant", [
        ("user",      "El técnico no se presentó a mi cita de instalación. Perdí medio día de trabajo."),
        ("assistant", "Lamento mucho lo ocurrido, entiendo su molestia y es completamente válida. Soy Carmen de Claro Perú. ¿Me da su DNI para revisar la orden de servicio?"),
        ("user",      "12378456."),
        ("assistant", "Verifico que la cita era a las 10am y el técnico no se presentó. Le ofrezco reagendar para mañana a la hora que prefiera y un crédito de S/30 en su cuenta como compensación."),
        ("user",      "De acuerdo, mañana a las 9am."),
        ("assistant", "Confirmado para mañana a las 9am. Recibirá un SMS de confirmación. Nuevamente disculpe el inconveniente. ¿Puedo ayudarle con algo más?"),
    ]),
    ("CONV-014", "In-Branch", "general_inquiry", "non_compliant", [
        ("user",      "¿Tienen planes de internet para zonas rurales en Cajamarca?"),
        ("assistant", "Para eso tiene que llamar al centro de atención."),
        ("user",      "¿No puede darme información básica aquí?"),
        ("assistant", "No tengo esa información disponible."),
    ]),
    ("CONV-015", "Call Center", "tech_support", "partial", [
        ("user",      "El WiFi solo llega a la mitad de mi departamento, el otro extremo no tiene señal."),
        ("assistant", "¿Dónde está ubicado su router actualmente?"),
        ("user",      "En la entrada del departamento."),
        ("assistant", "Pruebe moverlo al centro del apartamento. Si el problema persiste, podemos enviar un técnico."),
        ("user",      "¿El técnico tiene algún costo?"),
        ("assistant", "Depende del diagnóstico; en muchos casos es gratuito por garantía de servicio."),
    ]),
]

# ── Quality scores (pre-assigned per quality tier) ─────────────────────────

SCORES = {
    "CONV-001": ("yes",  "El agente siguió los 5 pasos: saludo, verificación, empatía, resolución y cierre.", 4.5, "Tono amable, solución concreta, cierre profesional.", "billing"),
    "CONV-002": ("no",   "No se presentó, no verificó identidad, no mostró empatía ni ofreció solución concreta.", 1.0, "Respuestas vagas, sin resolución, trato impaciente.", "tech_support"),
    "CONV-003": ("yes",  "Presentación, verificación de DNI, empatía implícita, solución clara y cierre correcto.", 5.0, "Solución completa, precio claro, agenda de instalación, excelente tono.", "plan_upgrade"),
    "CONV-004": ("no",   "No se presentó formalmente, no expresó empatía, escaló sin fecha precisa.", 2.5, "Resolvió parcialmente pero sin protocolo completo ni empatía visible.", "complaint"),
    "CONV-005": ("yes",  "Saludo, verificación, empatía, activación inmediata y cierre amigable.", 4.5, "Eficiente y amigable, solución inmediata, confirmación por SMS.", "plan_upgrade"),
    "CONV-006": ("no",   "No hubo saludo formal ni cierre; respuestas funcionales pero frías.", 2.5, "Información correcta pero atención impersonal sin protocolo.", "billing"),
    "CONV-007": ("no",   "Sin presentación, sin empatía, respuesta evasiva, sin resolución real.", 1.0, "Respuesta mínima y evasiva; no abordó el problema ni ofreció compensación.", "complaint"),
    "CONV-008": ("yes",  "Bienvenida, solicitud de datos, verificación, solución y cierre positivo.", 4.5, "Información completa, tono proactivo, guía al siguiente paso correctamente.", "general_inquiry"),
    "CONV-009": ("no",   "Sin saludo ni presentación, respuestas correctas pero protocolo incompleto.", 3.0, "Información útil y correcta, pero atención impersonal y sin cierre.", "plan_upgrade"),
    "CONV-010": ("no",   "Sin empatía, respuestas evasivas, no orientó al cliente sobre cómo resolver.", 1.0, "Trato brusco, sin solución ni orientación concreta.", "billing"),
    "CONV-011": ("yes",  "Presentación, verificación, empatía, resolución en tiempo y cierre profesional.", 5.0, "Servicio impecable, rapidez, trato empático, explicación de siguiente paso.", "tech_support"),
    "CONV-012": ("no",   "Redirigió sin presentarse ni verificar identidad; entregó folleto sin personalizar.", 2.5, "Atención básica sin protocolo; orientación mínima al cliente.", "plan_upgrade"),
    "CONV-013": ("yes",  "Empatía inmediata, verificación, compensación proactiva, reagendamiento y cierre.", 5.0, "Manejo ejemplar de queja: disculpa, compensación y solución concreta.", "complaint"),
    "CONV-014": ("no",   "Derivó sin información, sin empatía, sin alternativas de autoservicio.", 1.0, "Sin valor agregado; cliente se va sin ninguna información útil.", "general_inquiry"),
    "CONV-015": ("no",   "Sin presentación ni verificación; consejo técnico útil pero protocolo incompleto.", 3.0, "Consejo técnico adecuado pero atención mecánica sin calidez.", "tech_support"),
}

# ── Build DataFrames ────────────────────────────────────────────────────────

import pandas as pd

conv_rows   = []
scores_rows = []

for idx, (conv_id, channel, scenario, quality_tier, turns) in enumerate(CONVERSATIONS):
    ts = _BASE_TS + timedelta(days=idx * 4, hours=idx % 5)
    for turn_num, (role, content) in enumerate(turns, start=1):
        conv_rows.append({
            "conv_id":      conv_id,
            "channel":      channel,
            "scenario":     scenario,
            "quality_tier": quality_tier,
            "turn_num":     turn_num,
            "role":         role,
            "content":      content,
            "created_at":   ts,
        })

    comp, comp_rat, qual_score, qual_rat, reason = SCORES[conv_id]
    reason_rationale = {
        "billing":         "El cliente contactó para resolver un problema de facturación.",
        "tech_support":    "El cliente reportó una falla técnica en su servicio.",
        "plan_upgrade":    "El cliente solicitó cambio o contratación de un plan.",
        "complaint":       "El cliente presentó una queja formal por incumplimiento.",
        "general_inquiry": "El cliente realizó una consulta de información general.",
    }[reason]
    scores_rows.append({
        "conv_id":                conv_id,
        "channel":                channel,
        "scenario":               scenario,
        "quality_tier":           quality_tier,
        "protocol_compliance":    comp,
        "compliance_rationale":   comp_rat,
        "service_quality_score":  qual_score,
        "quality_rationale":      qual_rat,
        "contact_reason":         reason,
        "reason_rationale":       reason_rationale,
        "created_at":             ts,
    })

conv_df   = pd.DataFrame(conv_rows)
scores_df = pd.DataFrame(scores_rows)

# ── Write to Delta ──────────────────────────────────────────────────────────

spark.createDataFrame(conv_df).write.mode("overwrite").saveAsTable(CONV_TABLE)
print(f"✅ {len(conv_rows)} conversation turns → {CONV_TABLE}")

spark.createDataFrame(scores_df).write.mode("overwrite").saveAsTable(SCORES_TABLE)
print(f"✅ {len(scores_rows)} score rows → {SCORES_TABLE}")

print(f"\nUser suffix: {_short}")
print(f"Conversations table: {CONV_TABLE}")
print(f"Scores table:        {SCORES_TABLE}")
