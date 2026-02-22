"""
Asegura que cada combinación sede+especialidad tenga al menos 2 doctores.
Agrega los doctores faltantes a doctores.json y regenera horarios.json.
"""
import json, os, random
from datetime import date, timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def _load(f): 
    with open(os.path.join(DATA_DIR, f), encoding="utf-8") as fp:
        return json.load(fp)

def _save(f, data):
    with open(os.path.join(DATA_DIR, f), "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)

# Nombres de muestra para generar doctores adicionales
NOMBRES_M = ["Marco", "Rodrigo", "Pablo", "César", "Álvaro", "Bruno", "Iван", "Sergio", "Tomás", "Nicolás"]
NOMBRES_F = ["Paola", "Carmen", "Elena", "Natalia", "Alejandra", "Daniela", "Fernanda", "Gisela", "Renata", "Ximena"]
APELLIDOS  = ["Castro Ríos", "Ponce Vega", "Llanos Ruiz", "Salas Mora", "Tello Neyra",
               "Bravo Huanca", "Cano Soto", "Lagos Díaz", "Meza Fuentes", "Ríos Palma"]

def generar_colegiatura(n):
    return f"CMP-{50000 + n}"

def main():
    doctores    = _load("doctores.json")
    sede_esp    = _load("sede_especialidades.json")

    # Conteo actual por (sede_id, especialidad_id)
    conteo = {}
    for d in doctores:
        key = (d["sede_id"], d["especialidad_id"])
        conteo[key] = conteo.get(key, 0) + 1

    nuevos = []
    doc_num = len(doctores) + 1
    random.seed(99)

    for se in sede_esp:
        key = (se["sede_id"], se["especialidad_id"])
        faltantes = max(0, 2 - conteo.get(key, 0))
        for _ in range(faltantes):
            es_mujer = random.random() > 0.5
            nombre   = random.choice(NOMBRES_F if es_mujer else NOMBRES_M)
            apellido = random.choice(APELLIDOS)
            nuevos.append({
                "id": f"doc-{doc_num:03d}",
                "nombres": nombre,
                "apellidos": apellido,
                "especialidad_id": se["especialidad_id"],
                "sede_id": se["sede_id"],
                "numero_colegiatura": generar_colegiatura(doc_num),
            })
            conteo[key] = conteo.get(key, 0) + 1
            doc_num += 1

    todos_doctores = doctores + nuevos
    _save("doctores.json", todos_doctores)
    print(f"✅ Doctores: {len(doctores)} originales + {len(nuevos)} nuevos = {len(todos_doctores)} total")

    # Regenerar horarios para TODOS los doctores con fechas 2026
    franjas = [
        ("08:00","09:00"), ("09:00","10:00"), ("10:00","11:00"),
        ("11:00","12:00"), ("12:00","13:00"),
        ("14:00","15:00"), ("15:00","16:00"), ("16:00","17:00"), ("17:00","18:00"),
    ]

    hoy = date.today()
    dias_habiles = []
    d = hoy + timedelta(days=1)
    while len(dias_habiles) < 14:
        if d.weekday() < 6:
            dias_habiles.append(d)
        d += timedelta(days=1)

    random.seed(42)
    horarios = []
    hor_num  = 1

    for doc in todos_doctores:
        for dia in dias_habiles:
            for inicio, fin in franjas:
                estado = "disponible" if random.random() > 0.4 else "ocupado"
                horarios.append({
                    "id": f"hor-{hor_num:05d}",
                    "doctor_id": doc["id"],
                    "fecha": dia.isoformat(),
                    "hora_inicio": inicio,
                    "hora_fin": fin,
                    "estado": estado,
                })
                hor_num += 1

    _save("horarios.json", horarios)
    disponibles = sum(1 for h in horarios if h["estado"] == "disponible")
    print(f"✅ Horarios: {len(horarios)} generados ({disponibles} disponibles)")
    print(f"📅 Rango: {dias_habiles[0]} → {dias_habiles[-1]}")

    # Verificar resultado: mínimo 2 por sede+esp
    conteo_final = {}
    for d in todos_doctores:
        key = (d["sede_id"], d["especialidad_id"])
        conteo_final[key] = conteo_final.get(key, 0) + 1
    menos_de_2 = [k for k, v in conteo_final.items() if v < 2]
    if menos_de_2:
        print(f"⚠️ Aún hay combinaciones con menos de 2: {menos_de_2}")
    else:
        print("✅ Todas las combinaciones sede+especialidad tienen ≥ 2 doctores")

if __name__ == "__main__":
    main()
