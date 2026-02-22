"""
MediAgent - Data access layer (reads from JSON files)

Para migrar a Supabase, solo se reemplazan las funciones de este archivo.
La interfaz (inputs/outputs) se mantiene igual.
"""
import json
import os
from typing import Optional
from datetime import date

# ── Cargar datos desde JSON ──
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _load(filename: str) -> list:
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(filename: str, data: list):
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════
# Funciones de consulta (equivalen a queries SQL)
# ══════════════════════════════════════════════


def get_paciente_by_id(paciente_id: str) -> Optional[dict]:
    """Obtiene un paciente por su ID."""
    pacientes = _load("pacientes.json")
    for p in pacientes:
        if p["id"] == paciente_id:
            return p
    return None


def get_paciente_by_correo(correo: str) -> Optional[dict]:
    """Obtiene un paciente por su correo."""
    pacientes = _load("pacientes.json")
    for p in pacientes:
        if p["correo"] == correo:
            return p
    return None


def get_especialidad_nombre(especialidad_id: str) -> str:
    """Obtiene el nombre de una especialidad por su ID."""
    especialidades = _load("especialidades.json")
    for e in especialidades:
        if e["id"] == especialidad_id:
            return e["nombre"]
    return "Desconocida"


def get_sedes_cercanas(distrito_paciente: str, especialidad_id: str) -> list:
    """
    Busca sedes que tengan la especialidad requerida, estén cercanas al
    distrito del paciente Y que tengan al menos un doctor con horarios
    disponibles a partir de hoy.

    Solo muestra sedes con disponibilidad real para evitar mostrar opciones
    que luego terminen en 'no hay doctores disponibles'.
    """
    sedes = _load("sedes.json")
    sede_esp = _load("sede_especialidades.json")
    doctores = _load("doctores.json")
    horarios = _load("horarios.json")

    hoy = date.today().isoformat()

    # IDs de sedes que tienen la especialidad
    sedes_con_esp = {
        se["sede_id"] for se in sede_esp
        if se["especialidad_id"] == especialidad_id
    }

    # Doctor IDs con horarios disponibles futuros (precalcular una vez)
    docs_con_horario_disponible = {
        h["doctor_id"] for h in horarios
        if h["estado"] == "disponible" and h["fecha"] >= hoy
    }

    # Doctores de cada sede+especialidad con disponibilidad real
    def sede_tiene_disponibilidad(sede_id: str) -> bool:
        docs_en_sede = [
            d for d in doctores
            if d["sede_id"] == sede_id and d["especialidad_id"] == especialidad_id
        ]
        return any(d["id"] in docs_con_horario_disponible for d in docs_en_sede)

    # Filtrar por distrito cercano Y disponibilidad real
    resultado = []
    for s in sedes:
        if s["id"] not in sedes_con_esp:
            continue
        if s["distrito"] != distrito_paciente and distrito_paciente not in s.get("distritos_cercanos", []):
            continue
        if sede_tiene_disponibilidad(s["id"]):
            resultado.append(s)

    return resultado


def get_doctores_con_horarios(
    sede_id: str,
    especialidad_id: str,
    fecha_desde: str = None,
    fecha_hasta: str = None,
) -> list:
    """
    Busca doctores de una sede+especialidad con sus horarios disponibles.
    
    Retorna lista de dicts:
    [
      {
        "doctor": {id, nombres, apellidos, numero_colegiatura},
        "horarios": [{id, fecha, hora_inicio, hora_fin}, ...]
      }
    ]
    
    Equivale a:
    SELECT d.*, h.* FROM doctores d
    JOIN horarios h ON d.id = h.doctor_id
    WHERE d.sede_id = :sede AND d.especialidad_id = :esp
      AND h.estado = 'disponible' AND h.fecha >= CURRENT_DATE
    ORDER BY d.apellidos, h.fecha, h.hora_inicio
    """
    doctores = _load("doctores.json")
    horarios = _load("horarios.json")
    
    # Filtrar doctores de esa sede y especialidad
    docs_filtrados = [
        d for d in doctores 
        if d["sede_id"] == sede_id and d["especialidad_id"] == especialidad_id
    ]
    
    desde = fecha_desde if fecha_desde else date.today().isoformat()

    resultado = []
    for doc in docs_filtrados:
        # Horarios disponibles dentro del rango solicitado
        hors = [
            h for h in horarios
            if h["doctor_id"] == doc["id"]
            and h["estado"] == "disponible"
            and h["fecha"] >= desde
            and (fecha_hasta is None or h["fecha"] <= fecha_hasta)
        ]
        # Ordenar por fecha y hora
        hors.sort(key=lambda x: (x["fecha"], x["hora_inicio"]))
        
        if hors:  # Solo incluir doctores con horarios disponibles
            resultado.append({
                "doctor": {
                    "id": doc["id"],
                    "nombres": doc["nombres"],
                    "apellidos": doc["apellidos"],
                    "numero_colegiatura": doc["numero_colegiatura"]
                },
                "horarios": [
                    {
                        "id": h["id"],
                        "fecha": h["fecha"],
                        "hora_inicio": h["hora_inicio"],
                        "hora_fin": h["hora_fin"]
                    }
                    for h in hors
                ]
            })
    
    return resultado


def get_horario_by_id(horario_id: str) -> Optional[dict]:
    """Obtiene un horario por su ID."""
    horarios = _load("horarios.json")
    for h in horarios:
        if h["id"] == horario_id:
            return h
    return None


def get_doctor_by_id(doctor_id: str) -> Optional[dict]:
    """Obtiene un doctor por su ID."""
    doctores = _load("doctores.json")
    for d in doctores:
        if d["id"] == doctor_id:
            return d
    return None


def get_sede_by_id(sede_id: str) -> Optional[dict]:
    """Obtiene una sede por su ID."""
    sedes = _load("sedes.json")
    for s in sedes:
        if s["id"] == sede_id:
            return s
    return None


def crear_cita(paciente_id: str, doctor_id: str, sede_id: str, horario_id: str) -> dict:
    """
    Crea una cita y marca el horario como ocupado.
    
    Equivale a:
    BEGIN;
      INSERT INTO citas (...) VALUES (...);
      UPDATE horarios SET estado = 'ocupado' WHERE id = :horario_id;
    COMMIT;
    """
    import uuid
    
    # Crear la cita
    cita = {
        "id": f"cita-{str(uuid.uuid4())[:8]}",
        "paciente_id": paciente_id,
        "doctor_id": doctor_id,
        "sede_id": sede_id,
        "horario_id": horario_id,
        "estado": "confirmada"
    }
    
    # Guardar cita
    citas = _load("citas.json")
    citas.append(cita)
    _save("citas.json", citas)
    
    # Actualizar horario a ocupado
    horarios = _load("horarios.json")
    for h in horarios:
        if h["id"] == horario_id:
            h["estado"] = "ocupado"
            break
    _save("horarios.json", horarios)
    
    return cita
