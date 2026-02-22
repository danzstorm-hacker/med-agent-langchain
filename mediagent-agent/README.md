# 🏥 MediAgent — Agente Inteligente de Citas Médicas

<div align="center">

**Agente conversacional con IA para agendar citas médicas**

*LangChain · LangGraph · Claude AI · Streamlit · Resend*

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-HITL-FF6B35?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

</div>

---

## 📋 Descripción

**MediAgent** es un agente conversacional impulsado por IA que permite a los pacientes agendar citas médicas de manera natural y fluida. El agente guía al paciente paso a paso a través de un flujo inteligente: desde la selección de sede hasta la confirmación por correo electrónico.

### ✨ Características principales

- 🤖 **IA Conversacional** — Interacción natural en español con Claude (Anthropic)
- 🔄 **Human-in-the-Loop** — El paciente controla cada decisión del flujo
- 🏥 **Sedes inteligentes** — Recomienda sedes cercanas al distrito del paciente con disponibilidad real
- 📅 **Horarios dinámicos** — Muestra solo slots disponibles, con opción de ver la próxima semana
- 📧 **Confirmación por email** — Envía un correo HTML profesional al confirmar la cita (Resend)
- 🎨 **Interfaz visual** — Frontend interactivo construido con Streamlit

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                        🎨 FRONTEND                              │
│                     Streamlit (UI Chat)                          │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │  Login   │→ │  Chat Widget │→ │  Mensajes + Selecciones  │   │
│  │ Paciente │  │  Interactivo │  │  (Sedes, Doctores, etc.) │   │
│  └──────────┘  └──────┬───────┘  └──────────────────────────┘   │
└────────────────────────┼────────────────────────────────────────┘
                         │ Human-in-the-Loop (interrupt/resume)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     🧠 AGENTE (LangGraph)                       │
│                                                                  │
│  ┌──────────────────┐    ┌───────────────────┐                  │
│  │ Claude Haiku 4.5 │    │ Claude Haiku 4.5  │                  │
│  │   (llm_chat)     │    │   (llm_parse)     │                  │
│  │  Conversación    │    │  Parsing rápido   │                  │
│  │  max_tokens=512  │    │  max_tokens=5     │                  │
│  └────────┬─────────┘    └────────┬──────────┘                  │
│           │                       │                              │
│  ┌────────▼───────────────────────▼──────────┐                  │
│  │           State Machine (Grafo)           │                  │
│  │                                           │                  │
│  │  START                                    │                  │
│  │    │                                      │                  │
│  │    ▼                                      │                  │
│  │  ┌─────────────────────┐                  │                  │
│  │  │ 1. Clasificar       │ ◄── interrupt()  │                  │
│  │  │    + Sugerir Sedes   │     (elige sede) │                  │
│  │  └─────────┬───────────┘                  │                  │
│  │            │                               │                  │
│  │            ▼                               │                  │
│  │  ┌─────────────────────┐                  │                  │
│  │  │ 2. Doctores         │ ◄── interrupt()  │                  │
│  │  │    + Horarios        │     (elige doc)  │                  │
│  │  └─────────┬───────────┘                  │                  │
│  │            │                               │                  │
│  │            ▼                               │                  │
│  │  ┌─────────────────────┐                  │                  │
│  │  │ 3. Confirmar        │ ◄── interrupt()  │                  │
│  │  │    Resumen de cita   │     (sí/no)     │                  │
│  │  └─────────┬───────────┘                  │                  │
│  │            │                               │                  │
│  │            ▼                               │                  │
│  │  ┌─────────────────────┐                  │                  │
│  │  │ 4. Agendar          │                  │                  │
│  │  │    + Enviar Email 📧 │──► Resend API   │                  │
│  │  └─────────┬───────────┘                  │                  │
│  │            │                               │                  │
│  │            ▼                               │                  │
│  │          END ✅                            │                  │
│  └───────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     💾 CAPA DE DATOS                            │
│                                                                  │
│  tools.py (Data Access Layer)                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  get_paciente_by_id()    │  get_sedes_cercanas()        │   │
│  │  get_especialidad()      │  get_doctores_con_horarios()  │   │
│  │  get_doctor_by_id()      │  crear_cita()                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                         │                                        │
│              ┌──────────┴──────────┐                            │
│              ▼                     ▼                            │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │   JSON (Local)   │  │   Supabase (DB)  │                    │
│  │   📁 data/*.json │  │   🐘 PostgreSQL  │                    │
│  │   (MVP actual)   │  │   (Producción)   │                    │
│  └──────────────────┘  └──────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Flujo Conversacional Detallado

```
   PACIENTE                           MEDIAGENT                         SISTEMA
      │                                   │                                │
      │  "Hola, necesito una cita"        │                                │
      │──────────────────────────────────►│                                │
      │                                   │  get_sedes_cercanas()          │
      │                                   │───────────────────────────────►│
      │                                   │  ◄── sedes con disponibilidad  │
      │                                   │                                │
      │  🏥 "Estas son las sedes          │                                │
      │   cercanas a tu distrito:          │                                │
      │   1. Clínica San Pablo ⭐          │                                │
      │   2. Clínica Internacional"        │                                │
      │◄──────────────────────────────────│    ⏸️ interrupt()              │
      │                                   │                                │
      │  "La primera"                     │                                │
      │──────────────────────────────────►│  get_doctores_con_horarios()   │
      │                                   │───────────────────────────────►│
      │                                   │  ◄── doctores + slots          │
      │                                   │                                │
      │  👨‍⚕️ "Doctores disponibles:      │                                │
      │   Dr. Pérez - Lun 24: 09:00       │                                │
      │   Dra. López - Mar 25: 10:00"     │                                │
      │◄──────────────────────────────────│    ⏸️ interrupt()              │
      │                                   │                                │
      │  "Con el Dr. Pérez el lunes       │                                │
      │   a las 9"                        │                                │
      │──────────────────────────────────►│                                │
      │                                   │                                │
      │  📋 "Resumen de tu cita:           │                                │
      │   🏥 Clínica San Pablo            │                                │
      │   👨‍⚕️ Dr. Pérez                  │                                │
      │   📅 Lunes 24 feb, 09:00          │                                │
      │   ¿Confirmas? (sí/no)"            │                                │
      │◄──────────────────────────────────│    ⏸️ interrupt()              │
      │                                   │                                │
      │  "Sí, confirmo"                   │                                │
      │──────────────────────────────────►│  crear_cita()                  │
      │                                   │───────────────────────────────►│
      │                                   │  ◄── cita creada ✅            │
      │                                   │                                │
      │                                   │  enviar_correo_confirmacion()  │
      │                                   │───────────────────────────────►│ → Resend API
      │                                   │  ◄── email enviado 📧          │
      │                                   │                                │
      │  ✅ "¡Cita confirmada!            │                                │
      │   📧 Correo enviado a             │                                │
      │   tu@email.com"                   │                                │
      │◄──────────────────────────────────│                                │
      │                                   │                                │
```

---

## 📁 Estructura del Proyecto

```
mediagent-agent/
├── 📁 agent/                          # Núcleo del agente
│   ├── state.py                       # Estado (TypedDict) del grafo
│   ├── graph.py                       # Definición del grafo LangGraph
│   ├── nodes.py                       # 4 nodos: sedes → doctores → confirmar → agendar
│   ├── tools.py                       # Capa de acceso a datos (JSON/Supabase)
│   └── email_service.py              # Servicio de email con Resend
│
├── 📁 data/                           # Datos simulados (reemplazables por Supabase)
│   ├── especialidades.json            # 10 especialidades médicas
│   ├── sedes.json                     # 5 sedes en Lima
│   ├── sede_especialidades.json       # Relación sede ↔ especialidad
│   ├── doctores.json                  # 36 doctores
│   ├── horarios.json                  # ~1800 slots disponibles
│   ├── pacientes.json                 # 5 pacientes
│   └── citas.json                     # Citas creadas
│
├── 📁 scripts/                        # Utilidades de desarrollo
│   ├── agregar_doctores.py
│   ├── regenerar_horarios.py
│   ├── listar_modelos.py
│   └── verificar.py
│
├── main.py                            # Chat de consola (testing)
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🧠 Componentes Clave

### 1. LLM Dual — Velocidad optimizada

| Modelo | Uso | max_tokens | Propósito |
|---|---|---|---|
| `claude-haiku-4-5` (llm_chat) | Respuestas conversacionales | 512 | Generar mensajes amables y claros |
| `claude-haiku-4-5` (llm_parse) | Parsing de intención | 5 | Extraer número/opción del input (ultra rápido) |

### 2. Human-in-the-Loop (HITL)

Cada nodo usa `interrupt()` de LangGraph para pausar el grafo y esperar la decisión del paciente:

```python
# El grafo se PAUSA aquí y espera input del usuario
user_choice = interrupt({
    "message": "¿Cuál sede prefieres?",
    "type": "elegir_sede",
    "opciones": [...]
})
# El grafo RESUME cuando el paciente responde
```

### 3. Parseo inteligente de respuestas

El agente entiende respuestas naturales del paciente:

| Paciente dice | El agente entiende |
|---|---|
| `"1"` | Opción 1 (por número) |
| `"San Pablo"` | Sede por nombre (keyword match) |
| `"Con el Dr. Pérez el lunes a las 9"` | Doctor + día + hora (completo) |
| `"El martes"` | Solo día → pide doctor y hora |
| `"La próxima semana"` | Cambia rango de fechas |

### 4. Confirmación por Email (Resend)

Al confirmar la cita, se envía automáticamente un correo HTML profesional:

```
┌──────────────────────────────────────┐
│  🏥 MediAgent                        │
│  ──────────────────────────────────  │
│                                      │
│       ✅ Cita Confirmada             │
│                                      │
│  ¡Hola Andres! Tu cita ha sido      │
│  agendada exitosamente.              │
│                                      │
│  ┌────────────────────────────────┐  │
│  │ N° Cita:  cita-a1b2c3d4       │  │
│  │ 👨‍⚕️ Dr. Carlos Mendoza       │  │
│  │ 🩺 Cardiología                │  │
│  │ 📅 Lunes 24 de febrero        │  │
│  │ 🕐 09:00 - 09:30              │  │
│  │ 🏥 Clínica San Pablo          │  │
│  │ 📍 Av. El Polo 789, Surco     │  │
│  └────────────────────────────────┘  │
│                                      │
│  ⏰ Recuerda: Llegar 15 min antes   │
│  con tu DNI y exámenes previos.      │
│                                      │
└──────────────────────────────────────┘
```

---

## 🚀 Setup rápido

```bash
# 1. Clonar el repositorio
git clone https://github.com/danzstorm-hacker/med-agent-langchain.git
cd med-agent-langchain

# 2. Instalar dependencias
pip install -r mediagent-agent/requirements.txt

# 3. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus API keys:
#   ANTHROPIC_API_KEY=sk-ant-...
#   RESEND_API_KEY=re_...
#   EMAIL_FROM=MediAgent <noreply@tudominio.com>

# 4. Ejecutar (consola)
python mediagent-agent/main.py

# 5. Ejecutar (Streamlit)
streamlit run app.py
```

## 🧪 Pacientes de prueba

```bash
python main.py --paciente pac-001   # Andres Rojas (Miraflores, Cardiología)
python main.py --paciente pac-002   # Angy Anpat (San Borja, Dermatología)
python main.py --paciente pac-003   # Daniel Santos (Los Olivos, Traumatología)
python main.py --paciente pac-004   # Nahia Escalante (Surco, Ginecología)
python main.py --paciente pac-005   # Jhairo Yurivilca (San Isidro, Gastroenterología)
```

---

## 🛠️ Stack Tecnológico

| Componente | Tecnología | Rol |
|---|---|---|
| **LLM** | Claude Haiku 4.5 (Anthropic) | Generación de respuestas + parsing |
| **Orquestador** | LangGraph | State machine con HITL |
| **Framework** | LangChain | Integración con LLMs |
| **Frontend** | Streamlit | Interfaz de chat visual |
| **Email** | Resend | Envío de confirmaciones |
| **Datos (MVP)** | JSON local | Almacenamiento temporal |
| **Datos (Prod)** | Supabase (PostgreSQL) | Base de datos en producción |

---

## 🔀 Migración a Supabase

La capa de datos (`tools.py`) está diseñada para ser intercambiable. Para migrar a Supabase, solo se reemplazan las funciones sin tocar los nodos ni el grafo:

```python
# Antes (JSON local):
def get_sedes_cercanas(distrito, especialidad_id):
    sedes = _load("sedes.json")
    ...

# Después (Supabase):
def get_sedes_cercanas(distrito, especialidad_id):
    result = supabase.rpc("get_sedes_cercanas", {...}).execute()
    return result.data
```

---

<div align="center">

**Hecho con ❤️ por el equipo MediAgent**

*DataHackers Academy*

</div>
