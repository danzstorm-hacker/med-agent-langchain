"""
Regenera horarios.json con fechas desde hoy hacia adelante.
Ejecutar desde la carpeta mediagent-agent/: python scripts/regenerar_horarios.py
"""
import json
import os
import random
from datetime import date, timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def generar_horarios():
    # Cargar doctores
    with open(os.path.join(DATA_DIR, "doctores.json"), encoding="utf-8") as f:
        doctores = json.load(f)

    hoy = date.today()
    horarios = []
    horario_id = 1

    # Generar para los próximos 14 días hábiles (lun-sab)
    dias_habiles = []
    d = hoy + timedelta(days=1)  # empezar desde mañana
    while len(dias_habiles) < 14:
        if d.weekday() < 6:  # lunes(0) a sábado(5), excluir domingo(6)
            dias_habiles.append(d)
        d += timedelta(days=1)

    # Franjas horarias estándar (8am - 6pm)
    franjas = [
        ("08:00", "09:00"),
        ("09:00", "10:00"),
        ("10:00", "11:00"),
        ("11:00", "12:00"),
        ("12:00", "13:00"),
        ("14:00", "15:00"),
        ("15:00", "16:00"),
        ("16:00", "17:00"),
        ("17:00", "18:00"),
    ]

    random.seed(42)  # reproducible

    for doc in doctores:
        for dia in dias_habiles:
            for (inicio, fin) in franjas:
                # ~60% disponible, 40% ocupado — para hacerlo realista
                estado = "disponible" if random.random() > 0.4 else "ocupado"
                horarios.append({
                    "id": f"hor-{horario_id:05d}",
                    "doctor_id": doc["id"],
                    "fecha": dia.isoformat(),
                    "hora_inicio": inicio,
                    "hora_fin": fin,
                    "estado": estado,
                })
                horario_id += 1

    out_path = os.path.join(DATA_DIR, "horarios.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(horarios, f, ensure_ascii=False, indent=2)

    print(f"✅ Generados {len(horarios)} horarios para {len(doctores)} doctores")
    print(f"📅 Rango: {dias_habiles[0]} → {dias_habiles[-1]}")
    disponibles = sum(1 for h in horarios if h["estado"] == "disponible")
    print(f"📊 Disponibles: {disponibles} | Ocupados: {len(horarios) - disponibles}")

if __name__ == "__main__":
    generar_horarios()
