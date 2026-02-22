"""
MediAgent - Nodos del agente LangGraph

Cada función es un nodo del grafo. Los nodos que necesitan input del usuario
usan interrupt() para pausar el grafo y esperar la respuesta.
"""
from datetime import datetime
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

# ── LLM ──
llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0.3,
    max_tokens=1024,
)

SYSTEM_PROMPT = """Eres MediAgent, un asistente virtual médico amable y profesional.
Tu objetivo es ayudar a los pacientes a agendar citas médicas.
Responde siempre en español. Sé conciso, claro y usa un tono cálido.
Usa emojis médicos con moderación (🏥 👨‍⚕️ 📅 🕐 ✅) para hacer la conversación amigable.
NO inventes información. Solo usa los datos que se te proporcionan."""


def _format_fecha(fecha_str: str) -> str:
    """Convierte '2025-02-24' a 'Lunes 24 de febrero'."""
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


# ══════════════════════════════════════════════
# NODO 1: Clasificar intención + Sugerir sedes
# ══════════════════════════════════════════════

def nodo_clasificar_y_sedes(state: AgentState) -> dict:
    """
    Recibe el primer mensaje del paciente.
    Identifica la intención, saluda por nombre y muestra sedes cercanas.
    Luego PAUSA esperando que el paciente elija una sede.
    """
    paciente = state["paciente"]
    nombre = paciente["nombres"]
    especialidad = get_especialidad_nombre(paciente["especialidad_id"])
    distrito = paciente["distrito"]
    
    # Buscar sedes cercanas con la especialidad
    sedes = get_sedes_cercanas(distrito, paciente["especialidad_id"])
    
    if not sedes:
        msg = f"Lo siento {nombre}, no encontramos sedes cercanas a {distrito} con {especialidad}. 😔"
        return {
            "messages": [AIMessage(content=msg)],
            "etapa": "sin_sedes",
            "sedes_disponibles": [],
        }
    
    # Formatear opciones de sedes
    opciones_texto = "\n".join([
        f"  {i+1}. 🏥 {s['nombre']} — {s['direccion']} ({s['distrito']})"
        for i, s in enumerate(sedes)
    ])
    
    # Usar LLM para generar respuesta natural
    prompt = f"""El paciente {nombre} vive en {distrito} y necesita una consulta de {especialidad}.
Su mensaje fue: "{state['messages'][-1].content}"

Las sedes cercanas disponibles son:
{opciones_texto}

Genera una respuesta amigable que:
1. Salude al paciente por su nombre
2. Confirme que necesita {especialidad}
3. Muestre las sedes numeradas exactamente como se las paso
4. Pregunte cuál prefiere

IMPORTANTE: Muestra las sedes exactamente como están arriba, con sus números."""

    response = llm.invoke([
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
    
    # ── Después del resume: procesar elección ──
    sede_elegida = None
    
    # Intentar parsear por número
    try:
        num = int(user_choice.strip())
        if 1 <= num <= len(sedes):
            sede_elegida = sedes[num - 1]
    except (ValueError, AttributeError):
        pass
    
    # Si no fue número, buscar por nombre
    if not sede_elegida:
        for s in sedes:
            if any(keyword.lower() in user_choice.lower() for keyword in [
                s["nombre"], s["distrito"], s["nombre"].split()[-1]
            ]):
                sede_elegida = s
                break
    
    # Si aún no encontramos, usar LLM para interpretar
    if not sede_elegida:
        parse_prompt = f"""El paciente respondió: "{user_choice}"
Las opciones eran:
{opciones_texto}

¿Cuál sede eligió? Responde SOLO con el número (1, 2, etc). Si no es claro, responde "0"."""
        
        parse_response = llm.invoke([HumanMessage(content=parse_prompt)])
        try:
            num = int(parse_response.content.strip())
            if 1 <= num <= len(sedes):
                sede_elegida = sedes[num - 1]
        except ValueError:
            pass
    
    if not sede_elegida:
        sede_elegida = sedes[0]  # Fallback: primera opción
    
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
    Luego PAUSA esperando que el paciente elija doctor + horario.
    """
    paciente = state["paciente"]
    sede = state["sede_elegida"]
    especialidad = get_especialidad_nombre(paciente["especialidad_id"])
    
    # Buscar doctores con horarios
    doctores_hrs = get_doctores_con_horarios(sede["id"], paciente["especialidad_id"])
    
    if not doctores_hrs:
        msg = f"Lo siento, no hay doctores con horarios disponibles en {sede['nombre']} para {especialidad}. 😔"
        return {
            "messages": [AIMessage(content=msg)],
            "etapa": "sin_doctores",
            "doctores_horarios": [],
        }
    
    # Formatear doctores con horarios
    texto_doctores = ""
    opciones_flat = []  # Lista plana para facilitar selección
    opcion_num = 1
    
    for dh in doctores_hrs:
        doc = dh["doctor"]
        texto_doctores += f"\n👨‍⚕️ Dr(a). {doc['nombres']} {doc['apellidos']} ({doc['numero_colegiatura']})\n"
        
        agrupados = _agrupar_horarios_por_fecha(dh["horarios"])
        for fecha, horas in agrupados.items():
            fecha_fmt = _format_fecha(fecha)
            horas_fmt = ", ".join(horas)
            texto_doctores += f"   📅 {fecha_fmt}: {horas_fmt}\n"
        
        # Agregar a opciones planas
        for h in dh["horarios"]:
            opciones_flat.append({
                "numero": opcion_num,
                "doctor": doc,
                "horario": h,
                "texto": f"{doc['apellidos']} - {_format_fecha(h['fecha'])} {h['hora_inicio']}"
            })
            opcion_num += 1
    
    # Generar respuesta con LLM
    prompt = f"""El paciente eligió la sede {sede['nombre']}.
Especialidad: {especialidad}.
Estos son los doctores disponibles con sus horarios:

{texto_doctores}

Genera una respuesta que:
1. Confirme la sede elegida
2. Muestre los doctores con sus horarios exactamente como están arriba
3. Pida al paciente que elija un doctor y un horario específico (día y hora)

IMPORTANTE: Muestra los horarios exactamente como se proporcionan."""

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])
    
    agent_msg = response.content
    
    # ── HITL: Pausar y esperar elección de doctor + horario ──
    user_choice = interrupt({
        "message": agent_msg,
        "type": "elegir_doctor_horario",
        "doctores": doctores_hrs,
    })
    
    # ── Después del resume: parsear elección ──
    # Usar LLM para extraer doctor y horario de la respuesta libre
    opciones_texto = "\n".join([
        f"{o['numero']}. Dr(a). {o['doctor']['apellidos']} - {o['horario']['fecha']} {o['horario']['hora_inicio']}"
        for o in opciones_flat
    ])
    
    parse_prompt = f"""El paciente respondió: "{user_choice}"

Las opciones disponibles son:
{opciones_texto}

Identifica qué opción eligió el paciente. Responde SOLO con el número de la opción.
Si el paciente mencionó un doctor y un horario, busca la opción que coincida.
Si no es claro, responde con la opción más probable. Responde SOLO un número."""

    parse_response = llm.invoke([HumanMessage(content=parse_prompt)])
    
    doctor_elegido = None
    horario_elegido = None
    
    try:
        num = int(parse_response.content.strip())
        for o in opciones_flat:
            if o["numero"] == num:
                doctor_elegido = o["doctor"]
                horario_elegido = o["horario"]
                break
    except ValueError:
        pass
    
    # Fallback: primera opción del primer doctor
    if not doctor_elegido and opciones_flat:
        doctor_elegido = opciones_flat[0]["doctor"]
        horario_elegido = opciones_flat[0]["horario"]
    
    return {
        "messages": [
            AIMessage(content=agent_msg),
            HumanMessage(content=user_choice),
        ],
        "etapa": "doctor_elegido",
        "doctores_horarios": doctores_hrs,
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
    
    # Verificar confirmación
    respuesta = user_choice.strip().lower()
    confirmado = any(word in respuesta for word in ["sí", "si", "yes", "confirmo", "ok", "dale", "claro", "por supuesto"])
    
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
    
    # Crear la cita en BD
    cita = crear_cita(
        paciente_id=paciente["id"],
        doctor_id=doctor["id"],
        sede_id=sede["id"],
        horario_id=horario["id"],
    )
    
    # Mensaje de confirmación
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
