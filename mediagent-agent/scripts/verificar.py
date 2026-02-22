from agent.graph import graph
from agent.tools import get_sedes_cercanas, get_doctores_con_horarios

print("✅ Imports OK")

sedes = get_sedes_cercanas("Miraflores", "esp-002")
print(f"Sedes disponibles para pac-001 (Miraflores, Cardiología esp-002): {[s['nombre'] for s in sedes]}")

if sedes:
    docs = get_doctores_con_horarios(sedes[0]["id"], "esp-002")
    print(f"Doctores en {sedes[0]['nombre']}: {len(docs)} doctores con horarios")
    if docs:
        print(f"  Ejemplo: Dr. {docs[0]['doctor']['apellidos']} — {len(docs[0]['horarios'])} slots")
else:
    print("⚠️ No hay sedes disponibles")
