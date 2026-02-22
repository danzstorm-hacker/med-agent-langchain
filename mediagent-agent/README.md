# 🏥 MediAgent — Agente de Citas Médicas (MVP Local)

Agente conversacional para agendar citas médicas usando **LangChain + LangGraph** con patrón **Human-in-the-Loop**.

Esta versión funciona 100% local con datos JSON (sin Supabase). Ideal para iterar y probar la lógica del agente antes de integrar con base de datos y frontend.

## 🚀 Setup rápido

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Configurar API key
cp .env.example .env
# Editar .env y poner tu ANTHROPIC_API_KEY

# 3. Ejecutar
python main.py
```

## 🧪 Probar con diferentes pacientes

```bash
python main.py --paciente pac-001   # Daniel (Miraflores, Cardiología)
python main.py --paciente pac-002   # Sofía (San Borja, Dermatología)
python main.py --paciente pac-003   # Luis (Los Olivos, Traumatología)
python main.py --paciente pac-004   # Camila (Surco, Ginecología)
python main.py --paciente pac-005   # Javier (San Isidro, Gastroenterología)
```

## 📁 Estructura

```
mediagent-agent/
├── data/                        # Datos simulados (reemplazables por Supabase)
│   ├── especialidades.json      # 10 especialidades
│   ├── sedes.json               # 5 sedes en Lima
│   ├── sede_especialidades.json # Qué especialidades tiene cada sede
│   ├── doctores.json            # 36 doctores
│   ├── horarios.json            # ~1800 slots (Lun 24 Feb - Sáb 01 Mar)
│   ├── pacientes.json           # 5 pacientes de prueba
│   └── citas.json               # Citas creadas (empieza vacío)
├── agent/
│   ├── state.py                 # Estado del agente (TypedDict)
│   ├── tools.py                 # Acceso a datos (lee de JSON)
│   ├── nodes.py                 # Nodos del grafo (lógica + LLM)
│   └── graph.py                 # Definición del grafo LangGraph
├── main.py                      # Chat de consola
├── requirements.txt
└── .env.example
```

## 🔄 Flujo del agente

```
Paciente escribe → [clasificar_y_sedes] → HITL: elige sede
                 → [doctores_horarios]  → HITL: elige doctor+horario
                 → [confirmar]          → HITL: confirma sí/no
                 → [agendar]            → Cita creada ✅
```

**3 pausas Human-in-the-Loop:**
1. Elegir sede
2. Elegir doctor + horario (juntos)
3. Confirmar cita

## 🔀 Migrar a Supabase

Cuando el agente funcione bien, solo necesitas modificar `agent/tools.py`:

```python
# Antes (JSON):
def get_sedes_cercanas(distrito, especialidad_id):
    sedes = json.load("sedes.json")
    ...

# Después (Supabase):
def get_sedes_cercanas(distrito, especialidad_id):
    result = supabase.rpc("get_sedes_cercanas", {...}).execute()
    return result.data
```

La interfaz de cada función (inputs/outputs) es la misma. Los nodos, el grafo y el main no cambian.
