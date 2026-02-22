"""
MediAgent - Nodos del agente LangGraph

Cada función es un nodo del grafo. Los nodos que necesitan input del usuario
usan interrupt() para pausar el grafo y esperar la respuesta.

Optimizaciones de velocidad:
  - Modelo: claude-3-haiku-20240307 (5x más rápido que Sonnet, ideal para este caso)
  - LLM dual: llm_chat (respuestas) vs llm_parse (parsing de intención, max_tokens=5)
  - get_sedes_cercanas ya filtra sedes con disponibilidad real
  - Flujo robusto: si no hay doctores en la sede elegida, ofrece alternativas
"""
from datetime import datetime, date, timedelta
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import interrupt

from agent.tools import (
    get_paciente_by_id,
    get_especialidad_nombre,
    get_sedes_cercanas,
    get_doctores_con_horarios,
    get_doctor_by_id,
    get_sede_by_id,
    get_horario_by_id,
    crear_cita,
)
from agent.state import AgentState

# ── LLMs ──────────────────────────────────────────────────────────────────────
# llm_chat: genera respuestas conversacionales — Haiku es más que suficiente
# y entre 3-5x más rápido que Sonnet para estas tareas
llm_chat = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0.3,
    max_tokens=512,
)

# llm_parse: solo extrae un número o sí/no — max_tokens mínimo = máxima velocidad
llm_parse = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0,
    max_tokens=5,
)

SYSTEM_PROMPT = """Eres MediAgent, un asistente virtual médico amable y profesional.
Tu objetivo es ayudar a los pacientes a agendar citas médicas.
Responde siempre en español. Sé conciso, claro y usa un tono cálido.
Usa emojis médicos con moderación (🏥 👨‍⚕️ 📅 🕐 ✅) para hacer la conversación amigable.
NO inventes información. Solo usa los datos que se te proporcionan."""


def _format_fecha(fecha_str: str) -> str:
    """Convierte '2026-02-24' a 'Lunes 24 de febrero'."""
    dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    meses = ["", "enero", "febrero", "marzo", "abril", "mayo", "junio",
             "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
    d = datetime.strptime(fecha_str, "%Y-%m-%d")
    return f"{dias[d.weekday()]} {d.day} de {meses[d.month]}"


def _agrupar_horarios_por_fecha(horarios: list) -> dict:
    """Agrupa horarios por fecha para mostrar de forma legible."""
    agrupados = {}
    for h in horarios:
        fecha = h["fecha"]
        if fecha not in agrupados:
            agrupados[fecha] = []
        agrupados[fecha].append(h["hora_inicio"])
    return agrupados


def _calcular_semanas() -> tuple:
    """
    Calcula los rangos de esta semana y la próxima a partir de mañana.
    Semana = lunes a sábado.
    Returns: ((desde_actual, hasta_actual), (desde_sig, hasta_sig)) como strings ISO.
    """
    hoy = date.today()
    manana = hoy + timedelta(days=1)
    # Lunes de la semana que contiene mañana
    lunes = manana - timedelta(days=manana.weekday())
    sabado = lunes + timedelta(days=5)
    semana_actual = (manana.isoformat(), sabado.isoformat())
    lunes_sig = lunes + timedelta(weeks=1)
    sabado_sig = lunes_sig + timedelta(days=5)
    semana_siguiente = (lunes_sig.isoformat(), sabado_sig.isoformat())
    return semana_actual, semana_siguiente


def _quiere_siguiente_semana(text: str) -> bool:
    """Detecta si el usuario prefiere ver horarios de la semana siguiente."""
    keywords = [
        "próxima", "proxima", "siguiente", "otra semana", "no me cuadra",
        "no puedo", "semana que viene", "otra fecha", "no funciona",
        "ninguna", "no me quedan", "no disponible", "no me viene",
    ]
    return any(k in text.lower() for k in keywords)


def _formatear_doctores(doctores_hrs: list) -> tuple:
    """
    Formatea el texto de doctores+horarios y construye opciones_flat.
    Returns: (texto_doctores: str, opciones_flat: list)
    """
    texto = ""
    opciones_flat = []
    n = 1
    for dh in doctores_hrs:
        doc = dh["doctor"]
        texto += f"\n\U0001f468\u200d\u2695\ufe0f Dr(a). {doc['nombres']} {doc['apellidos']}\n"
        agrupados = _agrupar_horarios_por_fecha(dh["horarios"])
        for fecha, horas in agrupados.items():
            horas_fmt = ", ".join(horas)
            texto += f"   \U0001f4c5 {_format_fecha(fecha)}: {horas_fmt}\n"
        for h in dh["horarios"]:
            opciones_flat.append({
                "numero": n,
                "doctor": doc,
                "horario": h,
                "texto": f"{doc['apellidos']} - {_format_fecha(h['fecha'])} {h['hora_inicio']}",
            })
            n += 1
    return texto, opciones_flat


def _parsear_sede(user_input: str, sedes: list) -> dict | None:
    """
    Intenta identificar la sede elegida.
    1. Por número (más rápido, sin LLM)
    2. Por nombre/distrito en el texto (sin LLM)
    3. Fallback al LLM parser (solo si los anteriores fallan)
    """
    txt = user_input.strip()

    # Intento 1: número directo
    try:
        num = int(txt)
        if 1 <= num <= len(sedes):
            return sedes[num - 1]
    except ValueError:
        pass

    # Intento 2: keyword match (sin LLM — más rápido)
    txt_lower = txt.lower()
    for s in sedes:
        keywords = [s["nombre"].lower(), s["distrito"].lower(), s["nombre"].split()[-1].lower()]
        if any(k in txt_lower for k in keywords):
            return s

    # Intento 3: LLM parser con max_tokens=5
    opciones_txt = "\n".join([f"{i+1}. {s['nombre']} ({s['distrito']})" for i, s in enumerate(sedes)])
    parse_prompt = f"""El paciente respondió: "{user_input}"
Las opciones eran:
{opciones_txt}
¿Cuál sede eligió? Responde SOLO el número (1, 2, etc). Si no es claro responde 0."""

    resp = llm_parse.invoke([HumanMessage(content=parse_prompt)])
    try:
        num = int(resp.content.strip())
        if 1 <= num <= len(sedes):
            return sedes[num - 1]
    except ValueError:
        pass

    return None


def _parsear_opcion_numero(user_input: str, max_opcion: int, opciones_texto: str) -> int | None:
    """
    Parsea la opción elegida por número.
    1. Directo sin LLM
    2. Fallback LLM parser
    """
    try:
        num = int(user_input.strip())
        if 1 <= num <= max_opcion:
            return num
    except ValueError:
        pass

    # Fallback LLM
    parse_prompt = f"""El paciente respondió: "{user_input}"
Las opciones eran:
{opciones_texto}
¿Cuál opción eligió? Responde SOLO el número. Si no es claro responde 1."""
    resp = llm_parse.invoke([HumanMessage(content=parse_prompt)])
    try:
        num = int(resp.content.strip())
        if 1 <= num <= max_opcion:
            return num
    except ValueError:
        pass

    return None


# ══════════════════════════════════════════════
# NODO 1: Clasificar intención + Sugerir sedes
# ══════════════════════════════════════════════

def nodo_clasificar_y_sedes(state: AgentState) -> dict:
    """
    Recibe el primer mensaje del paciente.
    Muestra SOLO las sedes con disponibilidad real (filtradas en tools.py).
    Pausa esperando que el paciente elija una sede.
    """
    paciente = state["paciente"]
    nombre = paciente["nombres"]
    especialidad = get_especialidad_nombre(paciente["especialidad_id"])
    distrito = paciente["distrito"]

    # Sedes cercanas CON disponibilidad real (ya filtradas en get_sedes_cercanas)
    sedes = get_sedes_cercanas(distrito, paciente["especialidad_id"])

    if not sedes:
        msg = (
            f"Lo siento {nombre}, en este momento no encontramos sedes cercanas "
            f"a {distrito} con disponibilidad en {especialidad}. 😔\n"
            f"Te recomendamos llamar al 01-422-0000 para más opciones."
        )
        return {
            "messages": [AIMessage(content=msg)],
            "etapa": "sin_sedes",
            "sedes_disponibles": [],
        }

    # Formatear opciones
    opciones_texto = "\n".join([
        f"  {i+1}. 🏥 {s['nombre']} — {s['direccion']} ({s['distrito']})"
        for i, s in enumerate(sedes)
    ])

    prompt = f"""El paciente {nombre} vive en {distrito} y necesita una consulta de {especialidad}.
Su mensaje fue: "{state['messages'][-1].content}"

Las sedes disponibles (con doctores y horarios confirmados) son:
{opciones_texto}

Genera una respuesta amigable que:
1. Salude al paciente por su nombre
2. Confirme que necesita {especialidad}
3. Muestre las sedes numeradas exactamente como se las paso
4. Pregunte cuál prefiere

IMPORTANTE: Muestra las sedes exactamente como están arriba, con sus números."""

    response = llm_chat.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])
    agent_msg = response.content

    # ── HITL: Pausar y esperar elección de sede ──
    user_choice = interrupt({
        "message": agent_msg,
        "type": "elegir_sede",
        "opciones": [{"numero": i+1, "sede": s} for i, s in enumerate(sedes)],
    })

    # Parsear elección (rápido: número → keyword → LLM)
    sede_elegida = _parsear_sede(user_choice, sedes)
    if not sede_elegida:
        sede_elegida = sedes[0]  # fallback: primera opción

    return {
        "messages": [
            AIMessage(content=agent_msg),
            HumanMessage(content=user_choice),
        ],
        "etapa": "sede_elegida",
        "sedes_disponibles": sedes,
        "sede_elegida": sede_elegida,
    }


# ══════════════════════════════════════════════
# NODO 2: Mostrar doctores + horarios
# ══════════════════════════════════════════════

def nodo_doctores_horarios(state: AgentState) -> dict:
    """
    Muestra los doctores de la sede elegida con sus horarios disponibles.
    Si no hay doctores, ofrece al paciente elegir otra sede disponible.
    Pausa esperando que el paciente elija doctor + horario.
    """
    paciente = state["paciente"]
    sede = state["sede_elegida"]
    especialidad = get_especialidad_nombre(paciente["especialidad_id"])
    sedes_disponibles = state.get("sedes_disponibles", [])

    # Buscar doctores con horarios
    doctores_hrs = get_doctores_con_horarios(sede["id"], paciente["especialidad_id"])

    # ── Caso: no hay doctores en la sede elegida ──
    if not doctores_hrs:
        # Otras sedes disponibles (excluyendo la actual)
        otras_sedes = [s for s in sedes_disponibles if s["id"] != sede["id"]]

        if not otras_sedes:
            msg = (
                f"Lo siento, no hay disponibilidad en {sede['nombre']} para {especialidad} "
                f"y tampoco hay otras sedes cercanas disponibles. 😔\n"
                f"Te recomendamos llamar al 01-422-0000 para más opciones."
            )
            return {
                "messages": [AIMessage(content=msg)],
                "etapa": "sin_doctores",
                "doctores_horarios": [],
            }

        # Hay otras sedes: ofrecer alternativas
        opciones_texto = "\n".join([
            f"  {i+1}. 🏥 {s['nombre']} — {s['direccion']} ({s['distrito']})"
            for i, s in enumerate(otras_sedes)
        ])

        msg_alternativas = (
            f"Lo siento, en este momento no hay disponibilidad en **{sede['nombre']}** "
            f"para {especialidad}. 😔\n\n"
            f"Pero tenemos disponibilidad en estas otras sedes cercanas:\n\n"
            f"{opciones_texto}\n\n"
            f"¿Cuál de estas sedes prefieres? 😊"
        )

        # ── HITL: Pausar y esperar nueva elección ──
        user_choice = interrupt({
            "message": msg_alternativas,
            "type": "elegir_sede_alternativa",
            "opciones": [{"numero": i+1, "sede": s} for i, s in enumerate(otras_sedes)],
        })

        nueva_sede = _parsear_sede(user_choice, otras_sedes)
        if not nueva_sede:
            nueva_sede = otras_sedes[0]

        # Actualizar sede y buscar doctores en la nueva sede
        sede = nueva_sede
        doctores_hrs = get_doctores_con_horarios(sede["id"], paciente["especialidad_id"])

        if not doctores_hrs:
            msg = f"Parece que tampoco hay disponibilidad en {sede['nombre']} en este momento. 😔 Por favor llama al 01-422-0000."
            return {
                "messages": [
                    AIMessage(content=msg_alternativas),
                    HumanMessage(content=user_choice),
                    AIMessage(content=msg),
                ],
                "etapa": "sin_doctores",
                "sede_elegida": sede,
                "doctores_horarios": [],
            }

        # Continuar con la nueva sede
        return {
            "messages": [
                AIMessage(content=msg_alternativas),
                HumanMessage(content=user_choice),
            ],
            "etapa": "sede_elegida",
            "sede_elegida": sede,
            "doctores_horarios": doctores_hrs,
        }

    # ── Calcular rangos de esta semana y la próxima ──
    (desde_actual, hasta_actual), (desde_sig, hasta_sig) = _calcular_semanas()

    # Doctores disponibles ESTA SEMANA
    doctores_semana = get_doctores_con_horarios(
        sede["id"], paciente["especialidad_id"],
        fecha_desde=desde_actual, fecha_hasta=hasta_actual,
    )

    messages_extra = []
    doctores_para_mostrar = doctores_semana
    label_semana = "esta semana"

    # ── Si no hay slots esta semana → preguntar por la siguiente ──
    if not doctores_semana:
        msg_sin_semana = (
            f"No hay disponibilidad **esta semana** en {sede['nombre']} "
            f"para {especialidad}. \U0001f615\n\n"
            f"¿Te gustaría ver los horarios disponibles para la **próxima semana**? \U0001f4c5"
        )
        user_ans = interrupt({
            "message": msg_sin_semana,
            "type": "preguntar_siguiente_semana",
        })
        messages_extra += [AIMessage(content=msg_sin_semana), HumanMessage(content=user_ans)]

        confirmado = any(w in user_ans.lower() for w in [
            "sí", "si", "yes", "claro", "ok", "dale", "perfecto", "sólo", "solo", "quiero", "s"
        ])
        if not confirmado:
            msg_fin = "Entendido. Si cambias de opinión o necesitas otra fecha, con gusto te ayudamos. \U0001f60a"
            return {
                "messages": messages_extra + [AIMessage(content=msg_fin)],
                "etapa": "sin_doctores",
                "doctores_horarios": [],
            }

        # Cargar próxima semana
        doctores_semana_sig = get_doctores_con_horarios(
            sede["id"], paciente["especialidad_id"],
            fecha_desde=desde_sig, fecha_hasta=hasta_sig,
        )
        if not doctores_semana_sig:
            msg_fin = (
                f"Lo siento, tampoco hay disponibilidad la próxima semana en {sede['nombre']}. \U0001f615\n"
                f"Por favor llámanos al 01-422-0000 y te ayudamos a encontrar una fecha."
            )
            return {
                "messages": messages_extra + [AIMessage(content=msg_fin)],
                "etapa": "sin_doctores",
                "doctores_horarios": [],
            }

        doctores_para_mostrar = doctores_semana_sig
        label_semana = "la próxima semana"

    # ── Formatear y mostrar doctores de la semana elegida ──
    texto_drs, opciones_flat = _formatear_doctores(doctores_para_mostrar)

    prompt = f"""El paciente va a la sede {sede['nombre']} para {especialidad}.
Aquí están los doctores disponibles {label_semana}:

{texto_drs}

Genera una respuesta que:
1. Indique que estos son los horarios disponibles {label_semana}
2. Muestre exactamente los doctores y horarios como están arriba
3. {'Mencione que si ningún horario de esta semana le viene bien puede pedir ver la próxima semana' if label_semana == 'esta semana' else 'Pida elegir doctor, día y hora'}
4. Pida al paciente que elija doctor, día y hora

IMPORTANTE: Muestra los doctores y horarios exactamente como se presentan."""

    response = llm_chat.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])
    agent_msg = response.content

    # ── HITL: Pausar y esperar elección ──
    user_choice = interrupt({
        "message": agent_msg,
        "type": "elegir_doctor_horario",
        "doctores": doctores_para_mostrar,
    })
    messages_extra += [AIMessage(content=agent_msg), HumanMessage(content=user_choice)]

    # ── Detectar si el usuario pide la semana siguiente ──
    if label_semana == "esta semana" and _quiere_siguiente_semana(user_choice):
        doctores_semana_sig = get_doctores_con_horarios(
            sede["id"], paciente["especialidad_id"],
            fecha_desde=desde_sig, fecha_hasta=hasta_sig,
        )
        if not doctores_semana_sig:
            msg_no_sig = (
                f"Lo siento, tampoco hay disponibilidad la próxima semana en {sede['nombre']}. \U0001f615\n"
                f"Por favor llámanos al 01-422-0000."
            )
            return {
                "messages": messages_extra + [AIMessage(content=msg_no_sig)],
                "etapa": "sin_doctores",
                "doctores_horarios": [],
            }

        texto_sig, opciones_flat = _formatear_doctores(doctores_semana_sig)
        prompt_sig = f"""El paciente quiere ver horarios de la próxima semana en {sede['nombre']} para {especialidad}.
Aquí están los doctores disponibles la próxima semana:

{texto_sig}

Genera una respuesta amigable mostrando estos doctores y pidiendo que elija doctor, día y hora.
IMPORTANTE: Muestra los doctores y horarios exactamente como están arriba."""

        resp_sig = llm_chat.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt_sig),
        ])
        agent_msg_sig = resp_sig.content

        user_choice = interrupt({
            "message": agent_msg_sig,
            "type": "elegir_doctor_horario_semana_siguiente",
            "doctores": doctores_semana_sig,
        })
        messages_extra += [AIMessage(content=agent_msg_sig), HumanMessage(content=user_choice)]
        doctores_para_mostrar = doctores_semana_sig

    # ── Parsear selección de doctor + horario ──
    opciones_texto = "\n".join([
        f"{o['numero']}. Dr(a). {o['doctor']['apellidos']} - {o['horario']['fecha']} {o['horario']['hora_inicio']}"
        for o in opciones_flat
    ])
    num = _parsear_opcion_numero(user_choice, len(opciones_flat), opciones_texto)

    doctor_elegido = None
    horario_elegido = None

    if num:
        for o in opciones_flat:
            if o["numero"] == num:
                doctor_elegido = o["doctor"]
                horario_elegido = o["horario"]
                break

    # ── Si no hubo hora específica: detectar doctor+día y preguntar hora ──
    if not horario_elegido:
        doctor_detectado = None
        for dh in doctores_para_mostrar:
            apellido = dh["doctor"]["apellidos"].split()[0].lower()
            if apellido in user_choice.lower():
                doctor_detectado = dh
                break

        fecha_detectada = None
        if doctor_detectado:
            for fecha in {h["fecha"] for h in doctor_detectado["horarios"]}:
                nombre_dia = _format_fecha(fecha).lower()
                partes = nombre_dia.split()
                if any(p in user_choice.lower() for p in partes if len(p) > 3):
                    fecha_detectada = fecha
                    break

        if doctor_detectado and fecha_detectada:
            horas_disponibles = [
                h for h in doctor_detectado["horarios"] if h["fecha"] == fecha_detectada
            ]
            horas_txt = "\n".join([
                f"  {i+1}. \U0001f550 {h['hora_inicio']} - {h['hora_fin']}"
                for i, h in enumerate(horas_disponibles)
            ])
            doc = doctor_detectado["doctor"]
            msg_hora = (
                f"Perfecto, elegiste al Dr(a). {doc['nombres']} {doc['apellidos']} "
                f"el {_format_fecha(fecha_detectada)}. \U0001f4c5\n\n"
                f"Estas son las horas disponibles ese día:\n\n"
                f"{horas_txt}\n\n"
                f"¿A qué hora prefieres tu cita? \U0001f550"
            )
            user_hora = interrupt({
                "message": msg_hora,
                "type": "elegir_hora",
                "horas": horas_disponibles,
            })
            messages_extra += [AIMessage(content=msg_hora), HumanMessage(content=user_hora)]

            hora_num = _parsear_opcion_numero(
                user_hora, len(horas_disponibles),
                "\n".join([f"{i+1}. {h['hora_inicio']}" for i, h in enumerate(horas_disponibles)])
            )
            if hora_num:
                horario_elegido = horas_disponibles[hora_num - 1]
                doctor_elegido = doc
            elif horas_disponibles:
                horario_elegido = horas_disponibles[0]
                doctor_elegido = doc

    # Fallback final
    if not doctor_elegido and opciones_flat:
        doctor_elegido = opciones_flat[0]["doctor"]
        horario_elegido = opciones_flat[0]["horario"]

    return {
        "messages": messages_extra,
        "etapa": "doctor_elegido",
        "doctores_horarios": doctores_para_mostrar,
        "doctor_elegido": doctor_elegido,
        "horario_elegido": horario_elegido,
    }


# ══════════════════════════════════════════════
# NODO 3: Confirmar cita
# ══════════════════════════════════════════════

def nodo_confirmar(state: AgentState) -> dict:
    """
    Muestra resumen de la cita y pide confirmación.
    PAUSA esperando confirmación del paciente.
    """
    paciente = state["paciente"]
    sede = state["sede_elegida"]
    doctor = state["doctor_elegido"]
    horario = state["horario_elegido"]
    especialidad = get_especialidad_nombre(paciente["especialidad_id"])
    fecha_fmt = _format_fecha(horario["fecha"])

    resumen = f"""📋 **Resumen de tu cita:**

🏥 **Sede:** {sede['nombre']}
📍 **Dirección:** {sede['direccion']}
👨‍⚕️ **Doctor:** Dr(a). {doctor['nombres']} {doctor['apellidos']}
🩺 **Especialidad:** {especialidad}
📅 **Fecha:** {fecha_fmt}
🕐 **Hora:** {horario['hora_inicio']} - {horario['hora_fin']}
👤 **Paciente:** {paciente['nombres']} {paciente['apellidos']}

¿Confirmas esta cita? (sí/no)"""

    # ── HITL: Pausar y esperar confirmación ──
    user_choice = interrupt({
        "message": resumen,
        "type": "confirmar_cita",
    })

    # Parsear confirmación (sin LLM — simple keyword match)
    respuesta = user_choice.strip().lower()
    confirmado = any(word in respuesta for word in [
        "sí", "si", "yes", "confirmo", "ok", "dale", "claro", "por supuesto", "s"
    ])

    if not confirmado:
        msg = "Entendido, la cita no fue agendada. ¿Hay algo más en lo que pueda ayudarte? 😊"
        return {
            "messages": [
                AIMessage(content=resumen),
                HumanMessage(content=user_choice),
                AIMessage(content=msg),
            ],
            "etapa": "cancelado",
        }

    return {
        "messages": [
            AIMessage(content=resumen),
            HumanMessage(content=user_choice),
        ],
        "etapa": "confirmado",
    }


# ══════════════════════════════════════════════
# NODO 4: Agendar cita
# ══════════════════════════════════════════════

def nodo_agendar(state: AgentState) -> dict:
    """
    Crea la cita en la BD, actualiza el horario y envía confirmación.
    """
    paciente = state["paciente"]
    sede = state["sede_elegida"]
    doctor = state["doctor_elegido"]
    horario = state["horario_elegido"]
    especialidad = get_especialidad_nombre(paciente["especialidad_id"])
    fecha_fmt = _format_fecha(horario["fecha"])

    # Crear la cita
    cita = crear_cita(
        paciente_id=paciente["id"],
        doctor_id=doctor["id"],
        sede_id=sede["id"],
        horario_id=horario["id"],
    )

    msg = f"""✅ ¡Tu cita ha sido confirmada exitosamente!

📌 **Número de cita:** {cita['id']}
🏥 {sede['nombre']} — {sede['direccion']}
👨‍⚕️ Dr(a). {doctor['nombres']} {doctor['apellidos']}
📅 {fecha_fmt} de {horario['hora_inicio']} a {horario['hora_fin']}

📧 Te enviaremos un correo de confirmación a {paciente['correo']}.

Recuerda llegar 15 minutos antes de tu cita. ¿Hay algo más en lo que pueda ayudarte? 😊"""

    return {
        "messages": [AIMessage(content=msg)],
        "etapa": "cita_agendada",
        "cita_creada": cita,
    }
